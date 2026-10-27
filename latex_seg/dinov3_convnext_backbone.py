"""
DINOv3 ConvNeXt backbone with Hi-SAM-style adapters, for detectron2.

Same layout as the official DINOv3 ConvNeXt (dinov3/models/convnext.py; ConvNeXt v1 blocks with
LayerScale), so the official state dict loads directly; Hugging Face checkpoints
(facebook/dinov3-convnext-*-pretrain-lvd1689m) are converted on load. As in Hi-SAM, the
pretrained weights are frozen and every block gets two trainable adapters:
    x = x + gamma * (MLP(h) + 0.5 * MLP_Adapter(h)),  h = LN(Space_Adapter(dwconv(x)))
Stage outputs res2-res5 (strides 4/8/16/32) feed a standard FPN.
"""
import os
import re
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

from .convnext_backbone import Adapter

SIZES = {  # name: (depths, dims)
    "tiny": ((3, 3, 9, 3), (96, 192, 384, 768)),
    "small": ((3, 3, 27, 3), (96, 192, 384, 768)),
    "base": ((3, 3, 27, 3), (128, 256, 512, 1024)),
    "large": ((3, 3, 27, 3), (192, 384, 768, 1536)),
}


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels of a [B, C, H, W] map (official "channels_first" LayerNorm)."""

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias,
                            self.eps).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.0, adapters=True, adapter_scale=0.5):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.drop_prob = drop_path
        self.adapters = adapters
        if adapters:
            self.Space_Adapter = Adapter(dim)
            self.MLP_Adapter = Adapter(dim, skip_connect=False)
            self.adapter_scale = adapter_scale

    def forward(self, x):
        y = self.dwconv(x).permute(0, 2, 3, 1)  # (N, H, W, C)
        if self.adapters:
            y = self.Space_Adapter(y)
        h = self.norm(y)
        y = self.pwconv2(self.act(self.pwconv1(h)))
        if self.adapters:
            y = y + self.adapter_scale * self.MLP_Adapter(h)
        y = (self.gamma * y).permute(0, 3, 1, 2)
        if self.training and self.drop_prob > 0:
            keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) >= self.drop_prob
            y = y * keep / (1 - self.drop_prob)
        return x + y


def _convert_hf_state_dict(sd):
    """Hugging Face DINOv3ConvNextModel keys -> official DINOv3 ConvNeXt keys."""
    out = {}
    for k, v in sd.items():
        k = re.sub(r"^model\.", "", k)
        m = re.match(r"stages\.(\d+)\.downsample_layers\.(\d+)\.(.*)", k)
        if m:
            k = f"downsample_layers.{m[1]}.{m[2]}.{m[3]}"
        k = re.sub(r"^stages\.(\d+)\.layers\.(\d+)\.", r"stages.\1.\2.", k)
        k = k.replace(".depthwise_conv.", ".dwconv.").replace(".layer_norm.", ".norm.")
        k = k.replace(".pointwise_conv1.", ".pwconv1.").replace(".pointwise_conv2.", ".pwconv2.")
        k = k.replace(".layer_scale_parameter", ".gamma")
        if k.startswith("layer_norm."):
            k = k.replace("layer_norm.", "norm.")
        out[k] = v
    return out


class DINOv3ConvNeXt(Backbone):
    def __init__(self, size="large", out_features=("res2", "res3", "res4", "res5"), drop_path_rate=0.0,
                 adapters=True, grad_checkpoint=False):
        super().__init__()
        depths, dims = SIZES[size]
        self.downsample_layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4), LayerNorm2d(dims[0], eps=1e-6))])
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6), nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)))
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            self.stages.append(nn.Sequential(*[Block(dims[i], dpr[cur + j], adapters) for j in range(depths[i])]))
            cur += depths[i]
        self.grad_checkpoint = grad_checkpoint
        self._out_features = list(out_features)
        self._out_feature_strides = {f"res{i + 2}": 2 ** (i + 2) for i in range(4)}
        self._out_feature_channels = {f"res{i + 2}": dims[i] for i in range(4)}

    @property
    def size_divisibility(self):
        return 32

    def load_pretrained(self, path):
        """`path`: official .pth, a .safetensors file, or a Hugging Face model directory."""
        if os.path.isdir(path):
            path = os.path.join(path, "model.safetensors")
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = _convert_hf_state_dict(load_file(path))
        else:
            sd = torch.load(path, map_location="cpu")
        sd = {k: v for k, v in sd.items() if not k.startswith(("norm.", "head."))}  # final pooled norm unused
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [k for k in missing if "Adapter" not in k]
        assert not missing and not unexpected, (missing, unexpected)

    def forward(self, x):
        outputs = {}
        for i in range(4):
            x = self.downsample_layers[i](x)
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                # per block; blocks with a text-injection hook run outside checkpoint (the hook reads the
                # per-forward text context, which is gone when the backward recomputes)
                for blk in self.stages[i]:
                    x = blk(x) if getattr(blk, "has_text_hook", False) else checkpoint(blk, x, use_reentrant=False)
            else:
                x = self.stages[i](x)
            name = f"res{i + 2}"
            if name in self._out_features:
                outputs[name] = x
        return outputs


@BACKBONE_REGISTRY.register()
def build_dinov3_convnext_fpn_backbone(cfg, input_shape: ShapeSpec):
    c = cfg.MODEL.DINOV3_CONVNEXT
    bottom_up = DINOv3ConvNeXt(c.SIZE, cfg.MODEL.FPN.IN_FEATURES, c.DROP_PATH_RATE, c.ADAPTERS, c.GRAD_CHECKPOINT)
    if c.WEIGHTS:
        bottom_up.load_pretrained(c.WEIGHTS)
    if c.FREEZE:  # Hi-SAM: train only the adapters
        for n, p in bottom_up.named_parameters():
            p.requires_grad = "Adapter" in n
    return FPN(
        bottom_up=bottom_up,
        in_features=cfg.MODEL.FPN.IN_FEATURES,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
