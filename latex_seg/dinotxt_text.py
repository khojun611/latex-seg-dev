"""dino.txt text encoder (DINOv3 ViT-L/16 dino.txt, "DINOv2 Meets Text", arXiv 2412.16334) for torch 2.1.

Re-implements dinov3/eval/text/text_transformer.py (causal 24-layer, 1280-d) + TextHead linear projection
(1280 -> 2048) so the official dinov3 package (torch >= 2.4) is not needed. Weights: the `text_model.*` keys of
dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth.

The 2048-d output is [CLS-aligned half | patch-aligned half]; the patch-aligned half is the one the official
zero-shot segmentation compares with the frozen backbone's patch tokens. Same BPE vocabulary, SOT/EOT and
truncation as OpenAI CLIP, so `clip.tokenize(..., truncate=True)` produces identical token ids.
"""
import torch
import torch.nn.functional as F
from torch import nn


class _Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).unbind(2)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class _Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = _Attention(dim, num_heads)
        self.ffn_norm = nn.LayerNorm(dim)
        self.feed_forward = _Mlp(dim, int(dim * ffn_ratio))

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.ffn_norm(x))


class _TextTransformer(nn.Module):
    def __init__(self, context_length=77, vocab_size=49408, dim=1280, num_heads=20, num_layers=24, ffn_ratio=4.0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.positional_embedding = nn.Parameter(torch.zeros(context_length, dim))
        self.blocks = nn.ModuleList(_Block(dim, num_heads, ffn_ratio) for _ in range(num_layers))
        self.ln_final = nn.LayerNorm(dim)

    def forward(self, tokens):
        x = self.token_embedding(tokens) + self.positional_embedding[: tokens.shape[1]]
        for blk in self.blocks:
            x = blk(x)
        return self.ln_final(x)


class _Head(nn.Module):
    def __init__(self, dim=1280, embed_dim=2048):
        super().__init__()
        self.linear_projection = nn.Linear(dim, embed_dim, bias=False)

    def forward(self, x):
        return self.linear_projection(x)


class DinoTxtTextEncoder(nn.Module):
    """Frozen dino.txt text tower -> per-token features.
    part: "patch" (patch-aligned half, 1024-d), "cls" (CLS-aligned half), or "all" (2048-d)."""

    def __init__(self, weights, part="patch"):
        super().__init__()
        assert part in ("patch", "cls", "all"), part
        self.part = part
        self.backbone = _TextTransformer()
        self.head = _Head()
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        sd = {k[len("text_model."):]: v for k, v in sd.items() if k.startswith("text_model.")}
        self.load_state_dict(sd, strict=True)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @property
    def out_dim(self):
        return 2048 if self.part == "all" else 1024

    def train(self, mode=True):  # always frozen / eval
        return super().train(False)

    def forward(self, tokens):
        x = self.head(self.backbone(tokens))  # [B, T, 2048]
        if self.part == "patch":
            x = x[..., 1024:]
        elif self.part == "cls":
            x = x[..., :1024]
        return x
