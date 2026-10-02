"""Small decoder-only transformer (pre-norm, learned positions, tied embeddings). torch is imported inside functions only."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class LMConfig:
    vocab_size: int = 4097
    n_layer: int = 4
    d_model: int = 256
    n_head: int = 4
    ctx: int = 256

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def build_model(cfg: LMConfig) -> Any:
    import torch
    from torch import nn
    from torch.nn import functional as F

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.d_model)
            self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
            self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
            self.out = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

        def forward(self, x: Any) -> Any:
            b, t, c = x.shape
            q, k, v = self.qkv(self.ln1(x)).split(c, dim=2)
            q, k, v = (z.view(b, t, cfg.n_head, c // cfg.n_head).transpose(1, 2) for z in (q, k, v))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.proj(y.transpose(1, 2).reshape(b, t, c))
            return x + self.out(F.gelu(self.fc(self.ln2(x))))

    class GPT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos = nn.Embedding(cfg.ctx, cfg.d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(cfg.n_layer)])
            self.lnf = nn.LayerNorm(cfg.d_model)
            self.apply(self._init)

        @staticmethod
        def _init(m: Any) -> None:
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        def forward(self, idx: Any, targets: Any = None) -> Any:
            t = idx.shape[1]
            x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
            for blk in self.blocks:
                x = blk(x)
            logits = self.lnf(x) @ self.tok.weight.T
            if targets is None:
                return logits
            return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        @torch.no_grad()
        def generate(self, ids: list[int], max_new_tokens: int, temperature: float = 0.8, top_k: int = 40,
                     stop_id: int | None = None) -> list[int]:
            was = self.training
            self.eval()
            out = list(ids)
            for _ in range(max_new_tokens):
                x = torch.tensor([out[-cfg.ctx:]], dtype=torch.long)
                logits = self(x)[0, -1]
                if temperature <= 0:
                    nxt = int(torch.argmax(logits))
                else:
                    logits = logits / temperature
                    if top_k and top_k < logits.numel():
                        kth = torch.topk(logits, top_k).values[-1]
                        logits = logits.masked_fill(logits < kth, float("-inf"))
                    nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
                out.append(nxt)
                if stop_id is not None and nxt == stop_id:
                    break
            self.train(was)
            return out

    return GPT()


def count_params(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))
