# Eval

import itertools
from collections import defaultdict
from typing import List, Literal, Optional, Dict, Any, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from gymnasium.vector import SyncVectorEnv

from src.data.env import SetupDarkRoom
from src.dt.model import DecisionTransformer
from src.data.generate_goals import max_episode_reward


def _sample_from_logits(
    logits: torch.Tensor,
    mode: Literal["mode", "sample"] = "mode",
    temperature: float = 1.0,
    topk: Optional[int] = None,
    topp: Optional[float] = None,
) -> torch.Tensor:
    """
    logits: [N, A] -> [N] 动作索引
    """
    if mode == "mode":
        return torch.argmax(logits, dim=-1)

    x = logits
    if temperature <= 0:
        return torch.argmax(x, dim=-1)
    x = x / temperature

    # Top-k
    if topk is not None and 0 < topk < x.shape[-1]:
        top_vals, top_idx = torch.topk(x, k=topk, dim=-1)
        mask = torch.full_like(x, fill_value=float("-inf"))
        x = mask.scatter(-1, top_idx, top_vals)

    # Nucleus (top-p)
    if topp is not None and 0.0 < topp < 1.0:
        sorted_logits, sorted_idx = torch.sort(x, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        keep = cumsum <= topp
        keep[..., 0] = True  # 至少保留一个
        mask = torch.full_like(sorted_logits, fill_value=float("-inf"))
        pruned = torch.where(keep, sorted_logits, mask)
        x = torch.full_like(x, fill_value=float("-inf"))
        x.scatter_(-1, sorted_idx, pruned)

    dist = torch.distributions.Categorical(logits=x)
    return dist.sample()


@torch.no_grad()
def evaluate_in_context(
    env_config: SetupDarkRoom,
    model: DecisionTransformer,
    goal_idxs: List[int],
    eval_episodes: int,
    device: torch.device,
    seed: Optional[int] = None,
    mode: Literal["mode", "sample"] = "mode",
    dt_rtg: Literal["desired", "buffer"] = "buffer",
    rtg_gamma: float = 1.0,
    # ==== 可选项（不传与旧逻辑一致） ====
    temperature: float = 1.0,
    topk: Optional[int] = None,
    topp: Optional[float] = None,
    success_threshold: float = 0.0,             # 成功阈值（默认 return>0 即成功）
    normalize_by_max_return: bool = False,      # 归一化回报（除以每个 goal 的上界）
    max_env_steps: Optional[int] = None,        # 最多交互步数（全局）
    early_terminate_on_success: bool = False,   # 成功一次后该 env 立即开始下一条 episode
    progress: bool = False,                     # 是否显示 tqdm
) -> Tuple[Dict[int, List[float]], Dict[str, Any]]:
    """
    AD: (state, action, reward) × L。
    DT:
      - dt_rtg="buffer": 最近窗口 reward 的反向折扣和（与训练一致，推荐）。
      - dt_rtg="desired": max_return - prefix_sum（剩余目标回报）。

    流程：先用历史序列算 logits，再 step 环境，把“本步 action/reward”写回缓存；
    不会出现“用当前动作当条件”的泄露问题。读出位置由 model.dt_predict_from 决定。
    """

    # 一次性提示（仅 DT + action 槽读出时）
    try:
        if getattr(model, "paradigm", "ad") == "dt" and getattr(model, "dt_predict_from", "state") == "action":
            print("[note] DT reads from ACTION slots; training must use teacher-forcing. Eval here is safe.")
    except Exception:
        pass

    # 向量环境：每个 goal 一个 env
    vec_env = SyncVectorEnv([
        (lambda goal_idx=goal_idx: env_config.get_cls()(
            goal_index=goal_idx, enable_monitor_logs=False
        ).init_env())
        for goal_idx in goal_idxs
    ])

    T, N = model.seq_len, vec_env.num_envs

    # 序列缓存
    states  = torch.zeros((T, N), dtype=torch.long, device=device)
    actions = torch.zeros((T, N), dtype=torch.long, device=device)
    rewards = torch.zeros((T, N), dtype=torch.long, device=device)

    # 统计
    num_episodes = np.zeros(N, dtype=np.int64)
    returns = np.zeros(N, dtype=np.float32)
    ep_lengths = np.zeros(N, dtype=np.int32)
    successes = defaultdict(list)          # goal -> [0/1,...]
    episode_lengths = defaultdict(list)    # goal -> [len,...]
    eval_info: Dict[int, List[float]] = defaultdict(list)

    # 回报上界：用于 desired RTG 与可选归一化
    max_rtg_vec = torch.tensor(
        [max_episode_reward(g) for g in goal_idxs],
        dtype=torch.float32, device=device
    )  # [N]

    action_entropies: List[float] = []

    # 进度条（可关）
    pbar = None
    if progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=N * eval_episodes, position=1)
        except Exception:
            pbar = None

    is_dt = getattr(model, "paradigm", "ad") == "dt"

    state, _ = vec_env.reset(seed=seed)
    try:
        for step_glob in itertools.count(start=1):
            if max_env_steps is not None and step_glob > max_env_steps:
                break

            # 滚动窗口 & 写入当前观测
            states  = states.roll(-1, dims=0)
            actions = actions.roll(-1, dims=0)
            rewards = rewards.roll(-1, dims=0)
            states[-1] = torch.as_tensor(state, device=device, dtype=torch.long)

            win = min(step_glob, T)
            S = states[-win:].permute(1, 0)   # [N, win]
            A = actions[-win:].permute(1, 0)  # [N, win]
            R = rewards[-win:].permute(1, 0)  # [N, win]

            if is_dt:
                if dt_rtg == "buffer":
                    r = R.to(torch.float32)
                    if rtg_gamma == 1.0:
                        RTG = torch.flip(torch.cumsum(torch.flip(r, dims=[1]), dim=1), dims=[1])
                    else:
                        idx = torch.arange(win, device=device).view(1, win)
                        w = torch.pow(rtg_gamma, idx)
                        RTG = torch.flip(torch.cumsum(torch.flip(r * w, dims=[1]), dim=1), dims=[1]) / w
                    RTG = torch.clamp(RTG, min=0.0)
                elif dt_rtg == "desired":
                    r = R.to(torch.float32)
                    prefix = torch.cumsum(r, dim=1)
                    prefix = F.pad(prefix, (1, 0), value=0.0)[:, :-1]
                    cur = max_rtg_vec.unsqueeze(1) - prefix
                    RTG = torch.clamp(cur, min=0.0)
                else:
                    raise ValueError(f"Unknown dt_rtg: {dt_rtg}")

                logits = model(states=S, actions=A, rewards=R, rtg=RTG)[0][:, -1]  # [N, A]
            else:
                logits = model(states=S, actions=A, rewards=R)[0][:, -1]           # [N, A]

            # 动作熵（debug）
            probs = torch.softmax(logits, dim=-1)
            ent = torch.distributions.Categorical(probs=probs).entropy()
            action_entropies.append(ent.mean().item())

            # 采样 / 贪心
            act = _sample_from_logits(
                logits, mode=mode, temperature=temperature, topk=topk, topp=topp
            )

            # 环交互（SyncVectorEnv 默认对 done 自动 reset，并返回新 obs）
            state, reward, terminated, truncated, _ = vec_env.step(act.cpu().numpy())
            done = terminated | truncated

            # 写回缓存
            actions[-1] = act
            rewards[-1] = torch.as_tensor(reward, device=device, dtype=torch.long)

            # 累加统计
            returns += reward.astype(np.float32)
            ep_lengths += 1

            # 处理 episode 结束
            for i, d in enumerate(done):
                if d:
                    gidx = goal_idxs[i]
                    ret = returns[i]

                    ret_out = (float(ret / float(max_rtg_vec[i].item()))
                               if normalize_by_max_return and max_rtg_vec[i].item() > 0
                               else float(ret))

                    eval_info[gidx].append(ret_out)
                    episode_lengths[gidx].append(int(ep_lengths[i]))
                    successes[gidx].append(1.0 if ret > success_threshold else 0.0)

                    num_episodes[i] += 1
                    returns[i] = 0.0
                    ep_lengths[i] = 0

                    if pbar is not None:
                        pbar.update(1)

                    # 可选：成功立即进入下一条 episode（加快采样）
                    if (early_terminate_on_success and successes[gidx][-1] > 0.5
                            and num_episodes[i] < eval_episodes):
                        try:
                            state_i, _ = vec_env.envs[i].reset(seed=None)  # SyncVectorEnv 暴露 envs
                            state = np.array(state)
                            state[i] = state_i
                        except Exception:
                            pass

            if np.all(num_episodes >= eval_episodes):
                break

    finally:
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass
        try:
            vec_env.close()
        except Exception:
            pass

    # 便于外部直接读取的汇总指标
    per_goal_means = {g: float(np.mean(v)) for g, v in eval_info.items()} if eval_info else {}
    overall_mean = float(np.mean(list(per_goal_means.values()))) if per_goal_means else 0.0
    overall_success = float(np.mean([np.mean(v) for v in successes.values()])) if successes else 0.0

    debug_info: Dict[str, Any] = {
        "states": states,
        "actions": actions,
        "goal_idxs": goal_idxs,
        "action_entropy_mean": float(np.mean(action_entropies)) if action_entropies else None,
        "episode_lengths": dict(episode_lengths),
        "success_flags": dict(successes),
        "normalize_by_max_return": normalize_by_max_return,
        "success_threshold": success_threshold,
        "mode": mode,
        "dt_rtg": dt_rtg,
        "rtg_gamma": rtg_gamma,
        "temperature": temperature,
        "topk": topk,
        "topp": topp,
        "max_env_steps": max_env_steps,
        "early_terminate_on_success": early_terminate_on_success,
        # 新增汇总
        "per_goal_mean_return": per_goal_means,
        "overall_mean_return": overall_mean,
        "overall_success_rate": overall_success,
    }
    return eval_info, debug_info

