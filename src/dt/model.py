#Model

from typing import Optional, Literal, Tuple, List
import math
import torch
import torch.nn as nn
from torch import Tensor

# ---- Optional Mamba import (guarded) ----
_HAS_MAMBA = True
try:
    from mamba_ssm import Mamba  # pip install mamba-ssm
except Exception:
    _HAS_MAMBA = False


# ----------------- (kept for compatibility) Positional Encoding -----------------
# 不再用于主干，仅保留以便你需要恢复到 token-level 位置编码时可用
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.0, maxlen: int = 5000) -> None:
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)  # [1, maxlen, D]
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        L = token_embedding.size(1)
        return self.dropout(token_embedding + self.pos_embedding[:, :L, :])


# ----------------- Transformer Block (causal) -----------------
class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        ln_placem: Literal["postnorm", "prenorm"] = "postnorm",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ln_placem = ln_placem
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(residual_dropout),
        )
        # True = not allowed to attend
        self.register_buffer("causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool))
        self.seq_len = seq_len

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]
        if self.ln_placem == "postnorm":
            attn_out = self.attention(
                query=x, key=x, value=x, attn_mask=causal_mask,
                key_padding_mask=padding_mask, need_weights=False,
            )[0]
            x = self.norm1(x + self.drop(attn_out))
            x = self.norm2(x + self.mlp(x))
        else:
            nx = self.norm1(x)
            attn_out = self.attention(
                query=nx, key=nx, value=nx, attn_mask=causal_mask,
                key_padding_mask=padding_mask, need_weights=False,
            )[0]
            x = x + self.drop(attn_out)
            x = x + self.mlp(self.norm2(x))
        return x


# ----------------- Mamba Block (causal SSM) -----------------
class MambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        residual_dropout: float,
        ln_placem: Literal["postnorm", "prenorm"] = "postnorm",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        if not _HAS_MAMBA:
            raise ImportError(
                "Mamba backbone requested but mamba-ssm is not installed. "
                "Install via `pip install mamba-ssm`."
            )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ln_placem = ln_placem
        self.drop = nn.Dropout(residual_dropout)
        self.mamba = Mamba(d_model=hidden_dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(residual_dropout),
        )

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        if self.ln_placem == "postnorm":
            y = self.mamba(x)
            x = self.norm1(x + self.drop(y))
            x = self.norm2(x + self.mlp(x))
        else:
            y = self.mamba(self.norm1(x))
            x = x + self.drop(y)
            x = x + self.mlp(self.norm2(x))
        return x


# ----------------- Hybrid Block (Transformer <-> Mamba + gated fuse) -----------------
class HybridBlock(nn.Module):
    """
    Sandwich: Attn-MLP (residual) <-> Mamba-MLP (residual) -> gated fusion.
    通过 pattern 选择层次顺序："tm"=Transformer后接Mamba，"mt"=Mamba后接Transformer。
    gate 选择融合门控："none" | "linear" | "mlp"。
    """
    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        ln_placem: Literal["postnorm", "prenorm"] = "postnorm",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        # 新增：兼容 DecisionTransformer 传入的可选参数
        pattern: str = "tm",                                # "tm" or "mt"
        gate: Literal["none", "linear", "mlp"] = "mlp",
        **unused,                                           # 忽略将来可能新增的字段
    ):
        super().__init__()
        self.pattern = (pattern or "tm").lower()
        if self.pattern not in ("tm", "mt"):
            print(f"[HybridBlock] Unknown pattern={pattern!r}, fallback to 'tm'")
            self.pattern = "tm"

        # 两个子块（各自内部含残差/Norm/MLP）
        self.attn = TransformerBlock(
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            residual_dropout=residual_dropout,
            ln_placem=ln_placem,
        )
        self.mamba = MambaBlock(
            hidden_dim=hidden_dim,
            residual_dropout=residual_dropout,
            ln_placem=ln_placem,
            d_state=d_state, d_conv=d_conv, expand=expand,
        )

        # 门控方式
        self.gate_type = gate
        if gate == "none":
            self.gate = None
        elif gate == "linear":
            self.gate = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
        else:  # "mlp"（默认）
            self.gate = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        if self.pattern == "tm":
            u = self.attn(x, padding_mask=padding_mask)          # 先 Transformer
            v = self.mamba(u, padding_mask=padding_mask)          # 再 Mamba
        else:  # "mt"
            u = self.mamba(x, padding_mask=padding_mask)          # 先 Mamba
            v = self.attn(u, padding_mask=padding_mask)           # 再 Transformer

        if self.gate is None:                                     # 无门控：等权相加
            return 0.5 * (u + v)
        g = self.gate(torch.cat([u, v], dim=-1))                  # [B, T, D] ∈ (0,1)
        return g * v + (1.0 - g) * u



# ----------------- Unified DecisionTransformer -----------------
class DecisionTransformer(nn.Module):
    """
    - "ad": tokens per step = (state, action, reward)
    - "dt": tokens per step = (rtg, state, action)
      * 当 dt_predict_from="action" 时，会对动作做 **右移一位** 的 teacher-forcing：
        输入 a_prev = [<BOS>, a0, ..., a_{L-2}]，预测/监督目标是 a_{0..L-1}。
      * 位置编码采用 **per-timestep learnable embedding**：同一时间步的 (R,s,a)
        共享同一个时间步嵌入，而不是对 3L token 使用逐 token 的正弦位置编码。
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 10,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        ln_placem: Literal["postnorm", "prenorm"] = "postnorm",
        add_reward_head: bool = False,
        paradigm: Literal["ad", "dt"] = "ad",
        backbone: Literal["transformer", "mamba", "hybrid"] = "transformer",
        # ---- DT knobs ----
        rtg_proj: Literal["linear", "embedding"] = "linear",
        rtg_bins: int = 0,
        rtg_scale: float = 1.0,
        dt_predict_from: Literal["state", "action"] = "state",
        # ---- Mamba knobs (also as defaults for Hybrid 中的 Mamba) ----
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        # ---- Hybrid knobs (新加：用于吃掉/透传 config.yaml 字段) ----
        hybrid_pattern: str = "tmtm",                 # e.g. "tmtm", "ttmm"（如 HybridBlock 内部不使用也不报错）
        hybrid_gate: Literal["none", "linear", "mlp"] = "none",
        hybrid_mamba_d_state: Optional[int] = None,
        hybrid_mamba_d_conv: Optional[int] = None,
        hybrid_mamba_expand: Optional[int] = None,
        # ---- 兜底：忽略未来新增键，避免 unexpected-keyword 报错 ----
        **unused,
    ):
        super().__init__()
        self.paradigm = paradigm
        self.backbone = backbone
        self.dt_predict_from = dt_predict_from

        # token embeddings
        self.state_emb  = nn.Embedding(state_dim,  embedding_dim)
        self.action_emb = nn.Embedding(action_dim, embedding_dim)
        self.reward_emb = nn.Embedding(2,          embedding_dim)  # AD

        # DT: special BOS embedding for previous-action when using action readout
        self.action_bos = nn.Parameter(torch.zeros(embedding_dim))

        # RTG projection
        self.rtg_proj_type = rtg_proj
        self.rtg_bins = rtg_bins
        self.rtg_scale = rtg_scale
        if rtg_proj == "linear":
            assert rtg_scale > 0, "rtg_scale must be > 0"
            self.rtg_proj = nn.Linear(1, embedding_dim)
        else:
            assert rtg_bins > 1 and rtg_scale > 0, "rtg_bins>1 and rtg_scale>0 required"
            self.rtg_emb = nn.Embedding(rtg_bins, embedding_dim)

        # token type: 0=rtg/reward, 1=state, 2=action
        self.token_type_emb = nn.Embedding(3, embedding_dim)

        # ---- per-timestep learnable embedding ----
        self.seq_len = seq_len
        self.timestep_emb = nn.Embedding(seq_len, embedding_dim)
        self.embed_drop = nn.Dropout(embedding_dropout)

        # emb -> hid
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.emb2hid = nn.Linear(embedding_dim, hidden_dim) if embedding_dim != hidden_dim else None

        # ---- backbone stack over length 3*L (causal) ----
        L3 = 3 * seq_len
        blocks: List[nn.Module] = []

        # 若 hybrid_* 未指定，则回落到通用 mamba 超参
        eff_d_state  = hybrid_mamba_d_state  if hybrid_mamba_d_state  is not None else d_state
        eff_d_conv   = hybrid_mamba_d_conv   if hybrid_mamba_d_conv   is not None else d_conv
        eff_expand   = hybrid_mamba_expand   if hybrid_mamba_expand   is not None else expand

        if backbone == "transformer":
            for _ in range(num_layers):
                blocks.append(TransformerBlock(
                    seq_len=L3, hidden_dim=hidden_dim, num_heads=num_heads,
                    attention_dropout=attention_dropout, residual_dropout=residual_dropout,
                    ln_placem=ln_placem
                ))
        elif backbone == "mamba":
            if not _HAS_MAMBA:
                raise ImportError("Mamba backbone requested but mamba-ssm is not installed. Install via `pip install mamba-ssm`.")
            for _ in range(num_layers):
                blocks.append(MambaBlock(
                    hidden_dim=hidden_dim, residual_dropout=residual_dropout, ln_placem=ln_placem,
                    d_state=eff_d_state, d_conv=eff_d_conv, expand=eff_expand
                ))
        elif backbone == "hybrid":
            if not _HAS_MAMBA:
                raise ImportError("Hybrid backbone requires mamba-ssm installed. Install via `pip install mamba-ssm`.")
            # 如果你的 HybridBlock 支持 pattern/gate，就会用到；不支持也能正常构建（多余参数被忽略）
            for _ in range(num_layers):
                blocks.append(HybridBlock(
                    seq_len=L3, hidden_dim=hidden_dim, num_heads=num_heads,
                    attention_dropout=attention_dropout, residual_dropout=residual_dropout,
                    ln_placem=ln_placem,
                    # mamba 超参
                    d_state=eff_d_state, d_conv=eff_d_conv, expand=eff_expand,
                    # hybrid 额外控制
                    pattern=hybrid_pattern, gate=hybrid_gate,
                ))
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.blocks = nn.ModuleList(blocks)

        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.add_reward_head = add_reward_head
        if add_reward_head:
            self.reward_head = nn.Linear(hidden_dim * 2, 1)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.apply(self._init_weights)

    # ------------------------------------------------------------
    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    # ------------------------------------------------------------
    def _stack_triplets(self, parts: List[Tensor], type_ids: List[int]) -> Tensor:
        """parts: [B,L,D]×3  ->  [B,3L,D]；加 token-type 与 per-timestep 嵌入（每步共享）。"""
        assert len(parts) == 3 and len(type_ids) == 3
        B, L, D = parts[0].shape

        seq = torch.stack(parts, dim=2).contiguous().view(B, L * 3, D)

        type_tensor = torch.tensor(type_ids, device=seq.device).view(1, 1, 3).expand(B, L, 3).contiguous().view(B, L * 3)
        seq = seq + self.token_type_emb(type_tensor)

        step_ids = torch.arange(L, device=seq.device).view(1, L, 1).expand(B, L, 3).contiguous().view(B, L * 3)
        seq = seq + self.timestep_emb(step_ids)

        seq = self.embed_drop(seq)
        return seq

    def _tokenize_ad(self, states: Tensor, actions: Tensor, rewards: Tensor) -> Tuple[Tensor, slice, slice]:
        s = self.state_emb(states.long())
        a = self.action_emb(actions.long())
        r = self.reward_emb(rewards.long())
        seq = self._stack_triplets([s, a, r], type_ids=[1, 2, 0])  # (s,a,r)
        state_slice  = slice(0, None, 3)
        action_slice = slice(1, None, 3)
        return seq, state_slice, action_slice

    def _tokenize_dt(self, states: Tensor, actions: Tensor, rtg: Tensor) -> Tuple[Tensor, slice, slice]:
        # RtG -> embedding
        if self.rtg_proj_type == "linear":
            rtg_in = (rtg / self.rtg_scale).unsqueeze(-1).to(dtype=torch.float32)  # [B,L,1]
            r = self.rtg_proj(rtg_in)
        else:
            q = torch.clamp(torch.round(rtg / self.rtg_scale).long(), min=0, max=self.rtg_bins - 1)
            r = self.rtg_emb(q)

        s = self.state_emb(states.long())
        a_now = self.action_emb(actions.long())  # [B,L,D]

        # ---- teacher-forcing right shift when reading from action ----
        if self.dt_predict_from == "action":
            B, L, D = a_now.shape
            bos = self.action_bos.view(1, 1, D).expand(B, 1, D)       # [B,1,D]
            a_in = torch.cat([bos, a_now[:, :-1, :]], dim=1)          # [B,L,D] = [<BOS>, a0..a_{L-2}]
        else:
            a_in = a_now  # state readout: 不右移

        # order: (rtg, s, a_in) with token-types [0,1,2]
        seq = self._stack_triplets([r, s, a_in], type_ids=[0, 1, 2])
        state_slice  = slice(1, None, 3)
        action_slice = slice(2, None, 3)
        return seq, state_slice, action_slice

    # ------------------------------------------------------------
    def forward(
        self,
        states: Tensor,                        # [B, L]
        actions: Tensor,                       # [B, L]
        rewards: Tensor,                       # [B, L] (AD) – ignored for DT
        padding_mask: Optional[Tensor] = None, # [B, L] True = PAD
        rtg: Optional[Tensor] = None,          # [B, L] (DT only)
    ) -> Tuple[Tensor, Optional[Tensor]]:
        B, L = states.shape[0], states.shape[1]

        if self.paradigm == "ad":
            sequence, state_slice, action_slice = self._tokenize_ad(states, actions, rewards)
        elif self.paradigm == "dt":
            assert rtg is not None, "DT paradigm requires rtg tensor of shape [B, L]."
            sequence, state_slice, action_slice = self._tokenize_dt(states, actions, rtg)
        else:
            raise ValueError(f"Unknown paradigm: {self.paradigm}")

        if self.emb2hid is not None:
            sequence = self.emb2hid(sequence)

        # expand padding mask to 3L
        pad3 = None
        if padding_mask is not None:
            pad3 = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(B, 3 * L)
            ).to(torch.bool).contiguous()

        for block in self.blocks:
            sequence = block(sequence, padding_mask=pad3)

        # readout slice
        pred_slice = action_slice if (self.paradigm == "dt" and self.dt_predict_from == "action") else state_slice

        state_stream  = sequence[:, state_slice, :]
        action_stream = sequence[:, action_slice, :]

        action_logits = self.action_head(sequence[:, pred_slice, :])

        reward_logits = None
        if self.add_reward_head:
            reward_logits = self.reward_head(torch.cat([state_stream, action_stream], dim=-1)).squeeze(-1)

        return action_logits, reward_logits


