"""
Tryplicity — a decoder-only transformer built from scratch.

Architecture follows modern best practices:
  - RMSNorm (pre-norm)
  - Rotary Position Embeddings (RoPE)
  - Grouped Query Attention (GQA)
  - SwiGLU feed-forward
  - No bias terms
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
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys."""
    # q, k: (batch, n_heads, seq_len, head_dim)
    seq_len = q.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim//2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    def rotate(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_out = q * cos.repeat(1, 1, 1, 2) + rotate(q) * sin.repeat(1, 1, 1, 2)
    k_out = k * cos.repeat(1, 1, 1, 2) + rotate(k) * sin.repeat(1, 1, 1, 2)
    return q_out, k_out


class GQAttention(nn.Module):
    """Grouped Query Attention — fewer KV heads than query heads."""

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
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, cos, sin)

        # Expand KV heads to match query heads (GQA)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Try to use flash attention if available
        try:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None), dropout_p=self.dropout.p if self.training else 0.0)
        except RuntimeError:
            # Fallback to manual attention
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn + mask
            else:
                causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
                attn = attn + causal
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

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


class Tryplicity(nn.Module):
    """The Tryplicity language model."""

    def __init__(self, config: TryplicityConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying: share embedding and output weights
        self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE tables
        cos, sin = precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Scale residual projections by 1/sqrt(2*n_layers) for stability
        scale = (2 * self.config.n_layers) ** -0.5
        for layer in self.layers:
            nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=0.02 * scale)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)

        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
        """Autoregressive generation with top-k and top-p sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            idx = input_ids if input_ids.shape[1] <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.tok_emb.weight.numel()
        return n
