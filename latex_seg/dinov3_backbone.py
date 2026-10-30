"""
DINOv3 ViT backbone with Hi-SAM-style adapters, for detectron2.

A torch-2.1-compatible re-implementation of the DINOv3 ViT (RoPE, register tokens, LayerScale,
key-masked qkv bias) that loads the official checkpoints (e.g. dinov3_vitb16_pretrain_lvd1689m).
As in Hi-SAM, the pretrained weights are frozen and each block gets two trainable adapters:
    x = x + ls1(SpaceAdapter(attn(norm1(x))))
    x = x + ls2(mlp(norm2(x))) + 0.5 * MLPAdapter(norm2(x))
The stride-16 patch tokens are turned into p2-p6 by detectron2's SimpleFeaturePyramid (ViTDet).

MODEL.DINOV3.SPACE_ADAPTER_OUTSIDE_LS: DINOv3's LayerScale ls1 ranges ~0.05 (block 0) .. ~16 (block 23) in ViT-L, so
an adapter inside ls1 is nearly dead early and over-amplified late. With the flag the Space_Adapter residual is added
outside ls1 and zero-initialised:  x = x + ls1(attn) + SpaceAdapterDelta(attn).
MODEL.DINOV3.CONV_STEM: SFP's p2/p3 are deconvolutions of the stride-16 map (no sub-patch detail). The stem adds real
stride-4/8 features from the first two stages of the frozen DINOv3 ConvNeXt (same 7B teacher) to p2/p3 through
trainable zero-initialised projections, so training starts exactly at the plain SFP.
"""
import math
import os
import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone.vit import SimpleFeaturePyramid

ARCHS = {  # name: (embed_dim, depth, num_heads)
    "vits16": (384, 12, 6),
    "vitb16": (768, 12, 12),
    "vitl16": (1024, 24, 16),
}


class Adapter(nn.Module):
    """Hi-SAM / SAM-Adapter bottleneck: x (+) fc2(GELU(fc1(x)))."""

    def __init__(self, dim, mlp_ratio=0.25, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        hidden = int(dim * mlp_ratio)
        self.act = nn.GELU()
        self.D_fc1 = nn.Linear(dim, hidden)
        self.D_fc2 = nn.Linear(hidden, dim)

    def delta(self, x):
        return self.D_fc2(self.act(self.D_fc1(x)))

    def forward(self, x):
        xs = self.delta(x)
        return x + xs if self.skip_connect else xs


class RopePositionEmbedding(nn.Module):
    """DINOv3 2D RoPE (normalize_coords="separate"; train-time coordinate augmentation is not used)."""

    def __init__(self, embed_dim, num_heads, base=100.0):
        super().__init__()
        d_head = embed_dim // num_heads
        self.register_buffer(
            "periods", base ** (2 * torch.arange(d_head // 4, dtype=torch.float32) / (d_head // 2))
        )

    def forward(self, H, W):
        dd = dict(device=self.periods.device, dtype=torch.float32)
        coords_h = torch.arange(0.5, H, **dd) / H
        coords_w = torch.arange(0.5, W, **dd) / W
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
        coords = 2.0 * coords - 1.0                                                  # [HW, 2]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]      # [HW, 2, D/4]
        angles = angles.flatten(1, 2).tile(2)                                        # [HW, D]
        return torch.sin(angles), torch.cos(angles)


def _rope_apply(x, sin, cos):
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        # the key part of the qkv bias is masked out (DINOv3 mask_k_bias)
        self.qkv.register_buffer("bias_mask", torch.ones(dim * 3))
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = F.linear(x, self.qkv.weight, self.qkv.bias * self.qkv.bias_mask.to(self.qkv.bias.dtype))
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]                    # [B, h, N, d]
        sin, cos = rope
        prefix = N - sin.shape[-2]
        dtype = q.dtype
        q = torch.cat([q[:, :, :prefix], _rope_apply(q[:, :, prefix:].float(), sin, cos).to(dtype)], dim=2)
        k = torch.cat([k[:, :, :prefix], _rope_apply(k[:, :, prefix:].float(), sin, cos).to(dtype)], dim=2)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, adapters=True, adapter_scale=0.5, space_outside_ls=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = Mlp(dim, 4 * dim)
        self.ls2 = LayerScale(dim)
        self.adapters = adapters
        if adapters:
            self.Space_Adapter = Adapter(dim)
            self.MLP_Adapter = Adapter(dim, skip_connect=False)
            self.adapter_scale = adapter_scale
        self.space_outside_ls = adapters and space_outside_ls
        if self.space_outside_ls:  # starts as the exact pretrained block
            nn.init.zeros_(self.Space_Adapter.D_fc2.weight)
            nn.init.zeros_(self.Space_Adapter.D_fc2.bias)

    def forward(self, x, rope):
        a = self.attn(self.norm1(x), rope)
        if self.space_outside_ls:
            x = x + self.ls1(a) + self.Space_Adapter.delta(a)
        else:
            if self.adapters:
                a = self.Space_Adapter(a)
            x = x + self.ls1(a)
        h = self.norm2(x)
        x = x + self.ls2(self.mlp(h))
        if self.adapters:
            x = x + self.adapter_scale * self.MLP_Adapter(h)
        return x


class DINOv3ViT(Backbone):
    """Outputs the final-norm patch tokens as a stride-16 map {"last_feat": [B, C, H/16, W/16]}."""

    def __init__(self, arch="vitb16", n_storage_tokens=4, adapters=True, grad_checkpoint=False, space_outside_ls=False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        dim, depth, heads = ARCHS[arch]
        self.embed_dim = dim
        self.patch_size = 16
        self.n_prefix_tokens = 1 + n_storage_tokens
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, n_storage_tokens, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))  # unused, kept for strict loading
        self.rope_embed = RopePositionEmbedding(dim, heads)
        self.blocks = nn.ModuleList(Block(dim, heads, adapters, space_outside_ls=space_outside_ls) for _ in range(depth))
        for blk in self.blocks:
            blk.n_prefix_tokens = self.n_prefix_tokens  # read by the text-injection hooks
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self._out_features = ["last_feat"]
        self._out_feature_channels = {"last_feat": dim}
        self._out_feature_strides = {"last_feat": 16}

    def load_pretrained(self, path):
        if os.path.isdir(path) or path.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = _from_hf(load_file(os.path.join(path, "model.safetensors") if os.path.isdir(path) else path),
                          len(self.blocks), self.embed_dim)
            missing, unexpected = self.load_state_dict(sd, strict=False)
            # RoPE periods are recomputed (same formula); the k-bias mask is set by _from_hf
            missing = [k for k in missing if "Adapter" not in k and k != "rope_embed.periods"]
        else:
            sd = torch.load(path, map_location="cpu")
            missing, unexpected = self.load_state_dict(sd, strict=False)
            missing = [k for k in missing if "Adapter" not in k]
        assert not missing and not unexpected, (missing, unexpected)

    def forward(self, x):
        x = self.patch_embed.proj(x)
        B, C, H, W = x.shape
        x = torch.cat([self.cls_token.expand(B, -1, -1), self.storage_tokens.expand(B, -1, -1),
                       x.flatten(2).transpose(1, 2)], dim=1)
        rope = self.rope_embed(H, W)
        self.grid = (H, W)  # read by the text-injection / role-head hooks
        for blk in self.blocks:
            # blocks with a text-injection hook are not checkpointed: the hook reads the per-forward text
            # context, which is gone when the backward pass would recompute the block
            if self.grad_checkpoint and self.training and not getattr(blk, "has_text_hook", False):
                x = checkpoint(blk, x, rope, use_reentrant=False)
            else:
                x = blk(x, rope)
        x = self.norm(x)[:, self.n_prefix_tokens:]
        return {"last_feat": x.transpose(1, 2).reshape(B, C, H, W)}


def _from_hf(hf, depth, dim):
    """HF transformers DINOv3ViTModel state dict -> this module's (official) key layout."""
    sd = {
        "cls_token": hf["embeddings.cls_token"],
        "storage_tokens": hf["embeddings.register_tokens"],
        "mask_token": hf["embeddings.mask_token"].reshape(1, dim),
        "patch_embed.proj.weight": hf["embeddings.patch_embeddings.weight"],
        "patch_embed.proj.bias": hf["embeddings.patch_embeddings.bias"],
        "norm.weight": hf["norm.weight"],
        "norm.bias": hf["norm.bias"],
    }
    for i in range(depth):
        a, b = f"layer.{i}.", f"blocks.{i}."
        q, k, v = (hf[a + f"attention.{n}_proj.weight"] for n in "qkv")
        sd[b + "attn.qkv.weight"] = torch.cat([q, k, v], 0)
        zero = torch.zeros(dim, dtype=q.dtype)
        sd[b + "attn.qkv.bias"] = torch.cat([hf[a + "attention.q_proj.bias"], hf.get(a + "attention.k_proj.bias", zero),
                                             hf[a + "attention.v_proj.bias"]])
        sd[b + "attn.qkv.bias_mask"] = torch.cat([torch.ones(dim), torch.zeros(dim), torch.ones(dim)])
        sd[b + "attn.proj.weight"] = hf[a + "attention.o_proj.weight"]
        sd[b + "attn.proj.bias"] = hf[a + "attention.o_proj.bias"]
        for n in ("norm1", "norm2"):
            sd[b + n + ".weight"], sd[b + n + ".bias"] = hf[a + n + ".weight"], hf[a + n + ".bias"]
        sd[b + "ls1.gamma"] = hf[a + "layer_scale1.lambda1"]
        sd[b + "ls2.gamma"] = hf[a + "layer_scale2.lambda1"]
        sd[b + "mlp.fc1.weight"], sd[b + "mlp.fc1.bias"] = hf[a + "mlp.up_proj.weight"], hf[a + "mlp.up_proj.bias"]
        sd[b + "mlp.fc2.weight"], sd[b + "mlp.fc2.bias"] = hf[a + "mlp.down_proj.weight"], hf[a + "mlp.down_proj.bias"]
    return sd


class _SimpleFeaturePyramid(SimpleFeaturePyramid):
    """detectron2's SimpleFeaturePyramid spells the padding key "size_divisiblity", so ImageList
    never pads the input; expose the stride-32 divisibility properly."""

    @property
    def size_divisibility(self):
        return self._size_divisibility

    @property
    def padding_constraints(self):
        return {"size_divisibility": self._size_divisibility, "square_size": self._square_pad}


class ConvNeXtStem(nn.Module):
    """Frozen first two stages of the DINOv3 ConvNeXt -> {"p2": stride 4, "p3": stride 8} residuals for the SFP levels,
    through trainable 1x1 conv -> LN -> 3x3 conv projections whose last conv is zero-initialised."""

    STAGES = {"p2": 0, "p3": 1}

    def __init__(self, size, weights, out_channels, levels=("p2", "p3")):
        super().__init__()
        from .dinov3_convnext_backbone import DINOv3ConvNeXt, LayerNorm2d
        self.levels = tuple(levels)
        n = max(self.STAGES[l] for l in self.levels) + 1
        cnx = DINOv3ConvNeXt(size, out_features=(), adapters=False)
        if weights:
            cnx.load_pretrained(weights)
        self.downsample_layers = cnx.downsample_layers[:n]
        self.stages = cnx.stages[:n]
        for p in list(self.downsample_layers.parameters()) + list(self.stages.parameters()):
            p.requires_grad = False
        dims = [cnx._out_feature_channels[f"res{i + 2}"] for i in range(n)]
        self.proj = nn.ModuleDict()
        for l in self.levels:
            c = dims[self.STAGES[l]]
            last = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
            self.proj[l] = nn.Sequential(nn.Conv2d(c, out_channels, 1), LayerNorm2d(out_channels, eps=1e-6), last)

    def forward(self, x):
        feats = []
        with torch.no_grad():  # frozen
            for ds, st in zip(self.downsample_layers, self.stages):
                x = st(ds(x))
                feats.append(x)
        return {l: self.proj[l](feats[self.STAGES[l]]) for l in self.levels}


class _SimpleFeaturePyramidWithStem(_SimpleFeaturePyramid):
    def __init__(self, stem, **kwargs):
        super().__init__(**kwargs)
        self.stem = stem

    def forward(self, x):
        out = super().forward(x)
        for l, r in self.stem(x).items():
            out[l] = out[l] + r
        return out


@BACKBONE_REGISTRY.register()
def build_dinov3_vit_sfp_backbone(cfg, input_shape: ShapeSpec):
    c = cfg.MODEL.DINOV3
    net = DINOv3ViT(c.ARCH, adapters=c.ADAPTERS, grad_checkpoint=c.GRAD_CHECKPOINT,
                    space_outside_ls=c.SPACE_ADAPTER_OUTSIDE_LS)
    if c.WEIGHTS:
        net.load_pretrained(c.WEIGHTS)
    if c.FREEZE:  # Hi-SAM: train only the adapters (text-injection modules live outside the backbone)
        for n, p in net.named_parameters():
            p.requires_grad = "Adapter" in n
    sfp_kwargs = dict(
        net=net,
        in_feature="last_feat",
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        top_block=LastLevelMaxPool(),
        norm="LN",
    )
    s = c.CONV_STEM
    if s.ENABLED:
        stem = ConvNeXtStem(s.SIZE, s.WEIGHTS, cfg.MODEL.FPN.OUT_CHANNELS, s.LEVELS)
        return _SimpleFeaturePyramidWithStem(stem, **sfp_kwargs)
    return _SimpleFeaturePyramid(**sfp_kwargs)
