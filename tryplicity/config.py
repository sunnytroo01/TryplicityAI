import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TryplicityConfig:
    # Model architecture
    vocab_size: int = 32000
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4  # Grouped Query Attention: 4 KV heads, 16 query heads
    intermediate_size: int = 2816  # SwiGLU: ~2.7x dim (2/3 * 4 * dim)
    max_seq_len: int = 2048
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0

    # Efficiency: multi-token prediction (Meta, 2024)
    # Predict N future tokens simultaneously for 1.5-2x more learning signal
    n_future_tokens: int = 4  # 1 = standard next-token only, 4 = predict +1,+2,+3,+4

    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 8  # effective batch = 64 * 8 * 2048 = ~1M tokens
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    max_steps: int = 50000
    warmup_steps: int = 2000
    stable_steps: int = 40000  # WSD schedule: warmup -> stable -> decay
    grad_clip: float = 1.0
    seed: int = 42

    # Data
    data_dir: str = "/workspace/data/processed"
    tokenizer_path: str = "/workspace/tokenizer/tryplicity.model"

    # Checkpointing
    checkpoint_dir: str = "/workspace/checkpoints"
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 10

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def total_params(self) -> int:
        """Estimate total parameters."""
        emb = self.vocab_size * self.dim
        attn_per_layer = (
            self.dim * self.dim  # Q
            + self.dim * (self.dim // self.n_heads * self.n_kv_heads)  # K
            + self.dim * (self.dim // self.n_heads * self.n_kv_heads)  # V
            + self.dim * self.dim  # O
        )
        ffn_per_layer = 3 * self.dim * self.intermediate_size  # gate, up, down
        norm_per_layer = 2 * self.dim
        layer_total = attn_per_layer + ffn_per_layer + norm_per_layer
        total = 2 * emb + self.n_layers * layer_total + self.dim  # +final norm
        return total

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TryplicityConfig":
        with open(path) as f:
            return cls(**json.load(f))
