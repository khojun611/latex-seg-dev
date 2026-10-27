"""
ConvNeXt V2 (Hugging Face transformers) + FPN backbone for detectron2.

Stage outputs are exposed as res2-res5 (strides 4/8/16/32). Expects RGB input normalized
with ImageNet mean/std (set MODEL.PIXEL_MEAN / PIXEL_STD / INPUT.FORMAT accordingly).

With `adapters`, every block gets the two Hi-SAM adapters (train only them with `freeze`):
the Space_Adapter after the depthwise (spatial-mixing) conv and the MLP_Adapter in parallel
with the pointwise MLP, scaled by 0.5:
    x = x + drop_path(MLP(h) + 0.5 * MLP_Adapter(h)),  h = LN(Space_Adapter(dwconv(x)))
"""
import types
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import ConvNextV2Model

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool


class Adapter(nn.Module):
    """Hi-SAM / SAM-Adapter bottleneck on channels-last tensors: x (+) fc2(GELU(fc1(x)))."""

    def __init__(self, dim, mlp_ratio=0.25, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        hidden = int(dim * mlp_ratio)
        self.act = nn.GELU()
        self.D_fc1 = nn.Linear(dim, hidden)
        self.D_fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        xs = self.D_fc2(self.act(self.D_fc1(x)))
        return x + xs if self.skip_connect else xs


def _adapter_layer_forward(self, hidden_states):
    """ConvNextV2Layer.forward with the Hi-SAM adapters."""
    x = self.dwconv(hidden_states).permute(0, 2, 3, 1)  # (N, H, W, C)
    h = self.layernorm(self.Space_Adapter(x))
    x = self.pwconv2(self.grn(self.act(self.pwconv1(h))))
    x = x + self.adapter_scale * self.MLP_Adapter(h)
    return hidden_states + self.drop_path(x.permute(0, 3, 1, 2))


class ConvNeXtV2Backbone(Backbone):
    def __init__(self, weights, out_features=("res2", "res3", "res4", "res5"),
                 drop_path_rate=0.0, freeze=False, grad_checkpoint=False, adapters=False):
        super().__init__()
        # `weights`: Hugging Face model id or local directory
        self.model = ConvNextV2Model.from_pretrained(weights, drop_path_rate=drop_path_rate)
        self.model.layernorm = None  # final pooled-feature norm is not used
        if adapters:
            for stage in self.model.encoder.stages:
                for layer in stage.layers:
                    dim = layer.dwconv.in_channels
                    layer.Space_Adapter = Adapter(dim)
                    layer.MLP_Adapter = Adapter(dim, skip_connect=False)
                    layer.adapter_scale = 0.5
                    layer.forward = types.MethodType(_adapter_layer_forward, layer)
        if freeze:  # Hi-SAM: only the adapters (if any) stay trainable
            for n, p in self.model.named_parameters():
                p.requires_grad = "Adapter" in n
        self.grad_checkpoint = grad_checkpoint

        dims = self.model.config.hidden_sizes
        self._out_features = list(out_features)
        self._out_feature_strides = {f"res{i + 2}": 2 ** (i + 2) for i in range(4)}
        self._out_feature_channels = {f"res{i + 2}": dims[i] for i in range(4)}

    @property
    def size_divisibility(self):
        return 32

    def forward(self, x):
        outputs = {}
        x = self.model.embeddings(x)
        for i, stage in enumerate(self.model.encoder.stages):
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
            name = f"res{i + 2}"
            if name in self._out_features:
                outputs[name] = x
        return outputs


@BACKBONE_REGISTRY.register()
def build_convnextv2_fpn_backbone(cfg, input_shape: ShapeSpec):
    c = cfg.MODEL.CONVNEXT
    bottom_up = ConvNeXtV2Backbone(
        c.WEIGHTS,
        out_features=cfg.MODEL.FPN.IN_FEATURES,
        drop_path_rate=c.DROP_PATH_RATE,
        freeze=c.FREEZE,
        grad_checkpoint=c.GRAD_CHECKPOINT,
        adapters=c.ADAPTERS,
    )
    return FPN(
        bottom_up=bottom_up,
        in_features=cfg.MODEL.FPN.IN_FEATURES,
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
