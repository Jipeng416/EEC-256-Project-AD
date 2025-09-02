#train

import os
import tyro
from dataclasses import dataclass, asdict, field
import yaml
from typing import Tuple, Optional, List, Literal
from tqdm.auto import trange

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.nn import functional as F

# wandb 可选：未登录时自动禁用以避免报错
import wandb

from gymnasium.vector import SyncVectorEnv  # DT 在线评测
from src.data.env import SetupDarkRoom
from src.dt.seq_dataset import SequenceDataset
from src.dt.model import DecisionTransformer
from src.dt.schedule import cosine_annealing_with_warmup
from src.dt.eval import evaluate_in_context           # AD 评测
from src.data.generate_goals import max_episode_reward

DEVICE = os.getenv("DEVICE", "cpu")
if "cuda" in DEVICE:
    assert torch.cuda.is_available()


def get_goal_idxs(
    permutations_file: str = "saved_data/permutations_9.txt",
    train_test_split: float = 0.3,
    debug: bool = False,
):
    goal_idxs = np.loadtxt(permutations_file, dtype=int).tolist()
    test_size = int(len(goal_idxs) * train_test_split)
    train_idxs, test_idxs = goal_idxs[:-test_size], goal_idxs[-test_size:]
    if debug:
        train_idxs, test_idxs = train_idxs[:10], test_idxs[:10]
    return train_idxs, test_idxs


LEARNING_HISTORY_DIRS = [
    "saved_data/learning_history/ppo-01",
    "saved_data/learning_history/ppo-02",
    "saved_data/learning_history/ppo-03",
]


@dataclass
class TrainConfig:
    # ---- logging params ---
    env_config: SetupDarkRoom
    checkpoints_path: Optional[str] = "saved_data/saved_models"
    project: str = "AD"
    group: str = "debug"
    entity: str = "albinakl"

    # ---- dataset params ----
    permutations_file: str = "saved_data/permutations_9.txt"
    train_test_split: float = 0.5
    filter_episodes: int = 1
    learning_history_dirs: str | List[str] = field(
        default_factory=LEARNING_HISTORY_DIRS.copy
    )

    # ---- model params ----
    seq_len: int = 60
    embedding_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    ln_placem: Literal["postnorm", "prenorm"] = "postnorm"
    add_reward_head: bool = False
    load_from_checkpoint: Optional[str] = ""

    # 范式与骨干
    paradigm: Literal["ad", "dt"] = "ad"
    backbone: Literal["transformer", "mamba", "hybrid"] = "transformer"
    # ✅ 默认在“action 槽”读出（模型内部已做 teacher-forcing，避免标签泄露）
    dt_predict_from: Literal["state", "action"] = "action"

    # DT：RtG 表示与尺度
    rtg_proj: Literal["linear", "embedding"] = "linear"
    rtg_bins: int = 0
    rtg_scale: float = 1.0
    rtg_gamma: float = 1.0

    # Mamba/Hybrid 结构参数（会传入 DecisionTransformer）
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    # ---- optimizer params ----
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 5_000
    clip_grad: Optional[float] = 1.0

    # ---- dataloader params ----
    batch_size: int = 128
    num_updates: int = 300_000
    num_workers: int = 0

    # Hybrid 额外控制（顺序+门控）
    hybrid_pattern: Literal["tm", "mt"] = "tm"
    hybrid_gate: Literal["none", "linear", "mlp"] = "mlp"

    # ---- eval ----
    eval_freq: int = 1000
    eval_episodes: int = 10
    eval_seed: int = 0
    mode: Literal["mode", "sample"] = "mode"

    # ---- early stop（默认仅对 AD-Trans 开启触发条件）----
    early_stop_enable: bool = False
    early_stop_only_when: Literal["always", "ad-trans-only"] = "ad-trans-only"
    early_stop_metric: Literal["test_mean_return", "train_mean_return"] = "test_mean_return"
    early_stop_patience: int = 3
    early_stop_min_delta: float = 1e-4
    save_best: bool = True  # 评测更优时另存 best

    # ---- debug ----
    debug: bool = False

    def __post_init__(self):
        if self.checkpoints_path is not None:
            path = os.path.join(self.checkpoints_path, self.env_config.experiment_name)
            os.makedirs(path, exist_ok=True)
            self.checkpoints_path = path

    def save_args(self):
        config_file_path = os.path.join(self.checkpoints_path, "config.yaml")
        with open(config_file_path, "w") as config_file:
            config_file.write(yaml.safe_dump(asdict(self)))


def _init_wandb(config: TrainConfig):
    try:
        wandb.init(
            entity=config.entity,
            project=config.project,
            group=config.group,
            name=config.env_config.experiment_name,
            config=asdict(config),
        )
    except Exception:
        os.environ["WANDB_MODE"] = "disabled"
        wandb.init(
            mode="disabled",
            project=config.project,
            name=f"{config.env_config.experiment_name}-disabled",
            config=asdict(config),
        )


@torch.no_grad()
def _discounted_rtg(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """rewards: [B, L] -> RtG: [B, L]  with RtG_t = r_t + gamma * RtG_{t+1}"""
    B, L = rewards.shape
    rtg = torch.zeros((B, L), dtype=torch.float32, device=rewards.device)
    acc = torch.zeros((B,), dtype=torch.float32, device=rewards.device)
    for t in range(L - 1, -1, -1):
        acc = rewards[:, t].float() + gamma * acc
        rtg[:, t] = acc
    return rtg


@torch.no_grad()
def _evaluate_in_context_dt(env_config: SetupDarkRoom,
                            model: DecisionTransformer,
                            goal_idxs: List[int],
                            eval_episodes: int,
                            device: torch.device,
                            seed: int,
                            mode: Literal["mode", "sample"] = "mode"):
    """DT 在线评测：使用“剩余目标回报”（desired）作为 RtG。"""
    vec_env = SyncVectorEnv(
        [lambda goal_idx=goal_idx: env_config.get_cls()(
            goal_index=goal_idx, enable_monitor_logs=False
        ).init_env() for goal_idx in goal_idxs]
    )
    L = model.seq_len
    states  = torch.zeros((L, vec_env.num_envs), dtype=torch.long, device=device)
    actions = torch.zeros((L, vec_env.num_envs), dtype=torch.long, device=device)
    rewards = torch.zeros((L, vec_env.num_envs), dtype=torch.long, device=device)

    desired = np.array([max_episode_reward(i) for i in goal_idxs], dtype=np.float32)
    rtg_buf = torch.tensor(desired, device=device).repeat(L, 1)  # [L, N]

    import itertools
    from collections import defaultdict
    eval_info = defaultdict(list)
    num_episodes = np.zeros(vec_env.num_envs)
    returns = np.zeros(vec_env.num_envs)

    state, _ = vec_env.reset(seed=seed)
    for step in itertools.count(start=1):
        win = min(step, L)

        states  = states.roll(-1, dims=0)
        actions = actions.roll(-1, dims=0)
        rewards = rewards.roll(-1, dims=0)
        rtg_buf = rtg_buf.roll(-1, dims=0)

        states[-1] = torch.tensor(state, device=device)

        logits = model(
            states=states[-win:].permute(1, 0),
            actions=actions[-win:].permute(1, 0),
            rewards=rewards[-win:].permute(1, 0),
            rtg=rtg_buf[-win:].permute(1, 0),
        )[0][:, -1]

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.mode if mode == "mode" else dist.sample()

        state, reward, terminated, truncated, _ = vec_env.step(action.cpu().numpy())
        done = terminated | truncated

        actions[-1] = action
        rewards[-1] = torch.tensor(reward, device=device)

        desired = np.clip(desired - reward, a_min=0.0, a_max=None)
        rtg_buf[-1] = torch.tensor(desired, device=device)

        num_episodes += done.astype(int)
        returns += reward

        for i, d in enumerate(done):
            if d and num_episodes[i] <= eval_episodes:
                eval_info[goal_idxs[i]].append(returns[i])
                returns[i] = 0.0

        if np.all(num_episodes >= eval_episodes):
            break

    debug = {"states": states, "actions": actions, "goal_idxs": goal_idxs}
    return eval_info, debug


def _allow_early_stop(cfg: TrainConfig) -> bool:
    if not cfg.early_stop_enable:
        return False
    if cfg.early_stop_only_when == "always":
        return True
    # 默认只对 AD-Trans 开启
    return (cfg.paradigm == "ad" and cfg.backbone == "transformer")


def train(config: TrainConfig):
    config.save_args()
    _init_wandb(config)

    train_goal_idxs, test_goal_idxs = get_goal_idxs(
        permutations_file=config.permutations_file,
        train_test_split=config.train_test_split,
        debug=config.debug,
    )

    dataset = SequenceDataset(
        goal_idxs=train_goal_idxs,
        seq_len=config.seq_len,
        filter_episodes=config.filter_episodes,
        learning_history_dirs=config.learning_history_dirs,
    )

    # IterableDataset 不允许 shuffle/drop_last
    is_iterable = isinstance(dataset, IterableDataset) or not hasattr(dataset, "__len__")
    if is_iterable:
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            pin_memory=True,
            num_workers=config.num_workers,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            pin_memory=True,
            num_workers=config.num_workers,
            shuffle=True,
            drop_last=True,
        )
    data_iter = iter(dataloader)

    device = torch.device(DEVICE)

    # 临时环境用于维度确认
    tmp_env = config.env_config.init_env()
    model = DecisionTransformer(
        state_dim=tmp_env.observation_space.n,
        action_dim=tmp_env.action_space.n,
        seq_len=config.seq_len,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        ln_placem=config.ln_placem,
        add_reward_head=config.add_reward_head,
        paradigm=config.paradigm,
        backbone=config.backbone,
        rtg_proj=config.rtg_proj,
        rtg_bins=config.rtg_bins,
        rtg_scale=config.rtg_scale,
        dt_predict_from=config.dt_predict_from,  # 默认 "action"
        # 传入 Mamba/Hybrid 结构参数
        # 新增：Hybrid 参数
        hybrid_pattern=config.hybrid_pattern,
        hybrid_gate=config.hybrid_gate,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
    ).to(device)

    if config.load_from_checkpoint:
        model.load_state_dict(
            torch.load(config.load_from_checkpoint, map_location=device)
        )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = cosine_annealing_with_warmup(
        optimizer=optim,
        warmup_steps=config.warmup_steps,
        total_steps=config.num_updates,
    )

    # 早停跟踪
    use_es = _allow_early_stop(config)
    best_metric = float("-inf")
    bad_epochs = 0

    for step in trange(config.num_updates, desc="Training"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        states, actions, rewards = [b.to(device) for b in batch]

        if config.paradigm == "dt":
            rtg = _discounted_rtg(rewards, gamma=config.rtg_gamma)  # 模型内部会按 rtg_scale 归一
            action_logits, reward_logits = model(
                states=states, actions=actions, rewards=rewards, rtg=rtg
            )
        else:
            action_logits, reward_logits = model(
                states=states, actions=actions, rewards=rewards
            )

        # 交叉熵（标签仍然是原 actions；如走 action-readout，模型内部已做 teacher-forcing）
        loss_act = F.cross_entropy(
            action_logits.flatten(0, 1), actions.detach().flatten(0, 1)
        )

        # 可选 reward logits（二分类）
        if reward_logits is not None:
            loss_rew = F.binary_cross_entropy_with_logits(
                reward_logits.flatten(0, 1), rewards.detach().flatten(0, 1).float()
            )
        else:
            loss_rew = torch.tensor(0.0, device=device)

        loss = loss_act + loss_rew

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        with torch.no_grad():
            pred_a = torch.argmax(action_logits.flatten(0, 1), dim=-1)
            targ_a = actions.flatten()
            acc_a = (pred_a == targ_a).float().mean()

            log_dict = {
                "step": step,
                "lr": scheduler.get_last_lr()[0],
                "loss/total": loss.item(),
                "loss/action": loss_act.item(),
                "acc/action": acc_a.item(),
            }

            if reward_logits is not None:
                pred_r = (reward_logits.flatten() > 0).long()
                targ_r = rewards.flatten()
                acc_r = (pred_r == targ_r).float().mean()
                log_dict["acc/reward"] = acc_r.item()
                log_dict["loss/reward"] = loss_rew.item()

        wandb.log(log_dict, step=step)

        # 评测：AD 用原函数；DT 用在线 RtG 版本（desired）
        if (step % config.eval_freq == 0 or step == config.num_updates - 1):
            model.eval()
            if config.paradigm == "ad":
                eval_info_train, _ = evaluate_in_context(
                    config.env_config,
                    model,
                    train_goal_idxs,
                    config.eval_episodes,
                    device,
                    config.eval_seed,
                    mode=config.mode,
                )
                eval_info_test, _ = evaluate_in_context(
                    config.env_config,
                    model,
                    test_goal_idxs,
                    config.eval_episodes,
                    device,
                    config.eval_seed,
                    mode=config.mode,
                )
            else:
                eval_info_train, _ = _evaluate_in_context_dt(
                    config.env_config,
                    model,
                    train_goal_idxs,
                    config.eval_episodes,
                    device,
                    config.eval_seed,
                    mode=config.mode,
                )
                eval_info_test, _ = _evaluate_in_context_dt(
                    config.env_config,
                    model,
                    test_goal_idxs,
                    config.eval_episodes,
                    device,
                    config.eval_seed,
                    mode=config.mode,
                )
            model.train()

            train_mean = float(np.mean([h[-1] for h in eval_info_train.values()])) if eval_info_train else 0.0
            test_mean  = float(np.mean([h[-1] for h in eval_info_test.values()])) if eval_info_test else 0.0

            wandb.log(
                {
                    "eval/train_goals/mean_return": train_mean,
                    "eval/train_goals/median_return": float(np.median([h[-1] for h in eval_info_train.values()])) if eval_info_train else 0.0,
                    "eval/test_goals/mean_return": test_mean,
                    "eval/test_goals/median_return": float(np.median([h[-1] for h in eval_info_test.values()])) if eval_info_test else 0.0,
                    "epoch": step,
                },
                step=step,
            )

            # 早停判定（默认只对 AD-Trans）
            if use_es:
                metric = test_mean if config.early_stop_metric == "test_mean_return" else train_mean
                improved = (metric > best_metric + config.early_stop_min_delta)
                if improved:
                    best_metric = metric
                    bad_epochs = 0
                    if config.save_best and config.checkpoints_path is not None:
                        torch.save(model.state_dict(), os.path.join(config.checkpoints_path, "MODEL_best.pt"))
                        torch.save(optim.state_dict(), os.path.join(config.checkpoints_path, "OPTIM_best.pt"))
                    wandb.log({"early_stop/best_metric": best_metric}, step=step)
                else:
                    bad_epochs += 1
                    wandb.log({"early_stop/bad_epochs": bad_epochs}, step=step)
                    if bad_epochs >= config.early_stop_patience:
                        print(f"[EarlyStop] Stop at step={step} best={best_metric:.4f} metric={metric:.4f}")
                        break

            # 存档（常规快照）
            if config.checkpoints_path is not None:
                torch.save(
                    model.state_dict(),
                    os.path.join(config.checkpoints_path, f"model_{step}.pt"),
                )
                torch.save(
                    optim.state_dict(),
                    os.path.join(config.checkpoints_path, f"optim_{step}.pt"),
                )
                torch.save(
                    scheduler.state_dict(),
                    os.path.join(config.checkpoints_path, f"scheduler_{step}.pt"),
                )

    if config.checkpoints_path is not None:
        torch.save(
            model.state_dict(), os.path.join(config.checkpoints_path, "MODEL_last.pt")
        )


if __name__ == "__main__":
    tyro.cli(train)


