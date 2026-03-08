"""
Tryplicity — a decoder-only transformer built from scratch.

Architecture:
  - RMSNorm (pre-norm)
  - Rotary Position Embeddings (RoPE)
  - Grouped Query Attention (GQA) with QK-Norm
  - SwiGLU feed-forward
  - Multi-token prediction (auxiliary heads)
  - Z-loss for logit stability
  - Embedding gradient scaling (muP-lite)
  - No bias terms

Efficiency additions (Tier 2):
  - QK-Norm: normalizes Q and K before attention, allowing higher LR
    and preventing attention logit growth. Used by Cohere, Gemma 2.
  - Multi-token prediction: predict 2-4 future tokens simultaneously.
    Each forward pass extracts 2-4x more learning signal. (Meta, 2024)
  - Z-loss: small penalty on logit magnitude prevents FP16 overflow
    and improves training stability. (PaLM, Google)
  - Embedding scaling: scale embeddings by sqrt(dim) for muP-like
    behavior — better gradient flow to embeddings. (GPT-NeoX)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from tryplicity.config import TryplicityConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys."""
    seq_len = q.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    def rotate(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_out = q * cos.repeat(1, 1, 1, 2) + rotate(q) * sin.repeat(1, 1, 1, 2)
    k_out = k * cos.repeat(1, 1, 1, 2) + rotate(k) * sin.repeat(1, 1, 1, 2)
    return q_out, k_out


class GQAttention(nn.Module):
    """Grouped Query Attention with QK-Norm.

    QK-Norm (Dehghani et al., 2023): Apply RMSNorm to Q and K vectors
    before computing attention scores. This prevents attention logit
    growth as training progresses, which:
      - Allows 2-3x higher learning rates without instability
      - Prevents attention entropy collapse
      - Used by Cohere Command R, Gemma 2, Chameleon
    """

    def __init__(self, config: TryplicityConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        # QK-Norm: normalize queries and keys per-head
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm: normalize before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE after normalization
        q, k = apply_rope(q, k, cos, sin)

        # Expand KV heads for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # FlashAttention via SDPA
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            is_causal=(mask is None),
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: TryplicityConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Single transformer layer with pre-norm."""

    def __init__(self, config: TryplicityConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = GQAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MultiTokenPredictionHead(nn.Module):
    """Auxiliary head that predicts N future tokens simultaneously.

    From Meta (Gloeckle et al., 2024): Training with multi-token prediction
    extracts 2-4x more learning signal per forward pass. The model learns
    not just the next token but the next 2-4 tokens, forcing deeper
    understanding of text structure.

    Only the standard next-token head is used at inference.
    This is a pure training efficiency technique.
    """

    def __init__(self, config: TryplicityConfig):
        super().__init__()
        # Each auxiliary head has its own lightweight projection
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.proj = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(hidden))


class Tryplicity(nn.Module):
    """The Tryplicity language model."""

    def __init__(self, config: TryplicityConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Multi-token prediction: 3 auxiliary heads (predict tokens +2, +3, +4)
        self.n_future_tokens = config.n_future_tokens
        if self.n_future_tokens > 1:
            self.aux_heads = nn.ModuleList([
                MultiTokenPredictionHead(config)
                for _ in range(self.n_future_tokens - 1)
            ])
        else:
            self.aux_heads = None

        # Precompute RoPE
        cos, sin = precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Embedding scaling factor (muP-lite)
        self.emb_scale = math.sqrt(config.dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale residual projections
        scale = (2 * self.config.n_layers) ** -0.5
        for layer in self.layers:
            nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=0.02 * scale)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape

        # Embedding with muP-like scaling
        x = self.tok_emb(input_ids) * self.emb_scale

        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Primary next-token loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            # Z-loss: penalize large logits for numerical stability (PaLM)
            # Prevents FP16/BF16 overflow and encourages sharper distributions
            z_loss = 1e-4 * logits.float().logsumexp(dim=-1).pow(2).mean()
            loss = loss + z_loss

            # Multi-token prediction auxiliary losses
            if self.aux_heads is not None and T > self.n_future_tokens:
                for i, head in enumerate(self.aux_heads):
                    offset = i + 2  # head 0 predicts +2, head 1 predicts +3, etc.
                    if offset < T:
                        aux_logits = head(x[:, :-offset, :])
                        aux_targets = targets[:, offset:]
                        min_len = min(aux_logits.shape[1], aux_targets.shape[1])
                        aux_loss = F.cross_entropy(
                            aux_logits[:, :min_len].reshape(-1, aux_logits.size(-1)),
                            aux_targets[:, :min_len].reshape(-1),
                            ignore_index=-1,
                        )
                        # Auxiliary losses weighted lower (0.1 each)
                        loss = loss + 0.1 * aux_loss

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
        """Autoregressive generation with top-k and top-p sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            idx = input_ids if input_ids.shape[1] <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs_sorted = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs_sorted, dim=-1)
                remove = (cumulative - probs_sorted) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits)
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.tok_emb.weight.numel()
        return n
