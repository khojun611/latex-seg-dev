# Copyright (c) Facebook, Inc. and its affiliates.
"""
LaTeX-Seg: text-conditioned FPN segmentation.

Text injection uses the Hi-SAM residual form
    x = x + gate * CrossAttn(q = LayerNorm(x), k = v = Linear(text tokens))
with a learnable scalar gate (init 0.05); without text the injection is skipped (pure visual
path). Where it is applied is set by MODEL.TEXT_INJECTION.LOCATION:
  "backbone": after each of the last N blocks of the stride-16 stage (ResNet res4 /
      ConvNeXt stage 3 / the DINOv3 ViT blocks), semantic -> style -> location text in order.
  "fpn": on the FPN levels, coarse to fine as in the paper (Eq. 4), see ProgressiveTextFusion.

Coarse head: PanopticFPN scale heads summed at stride 4 -> SIREN glyph distance field (GDF)
predictor -> GDF cross-attention (gated, zero-init) -> 2x conv/BN -> 1x1 predictor.
Text tokens are per-token CLIP features (LoRA on the text transformer).
"""
from typing import Callable, Dict, Optional, Union
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
import clip
from peft import LoraConfig, get_peft_model

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import META_ARCH_REGISTRY, SEM_SEG_HEADS_REGISTRY, Backbone, build_backbone
from detectron2.modeling.meta_arch.semantic_seg import build_sem_seg_head
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

from .pv_heads import PVHeads, split_semantic_words
from .role_heads import RoleHeads


# ----------------------------------------------------------------------------
# Text injection (Hi-SAM style)
# ----------------------------------------------------------------------------
TEXT_ORDER = ("semantic", "style", "location")  # first -> last injected block


class TextInjection(nn.Module):
    """x = x + gate * CrossAttn(LayerNorm(x), Linear(text)).
    x is a [B, C, H, W] map, or [B, N, C] ViT tokens whose first `n_prefix` (cls/register)
    tokens are left untouched."""

    def __init__(self, dim, text_dim=512, num_heads=8, gate_init=0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.kv_proj = nn.Linear(text_dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x, text, pad_mask=None, n_prefix=0):
        tokens = x.dim() == 3
        if tokens:
            prefix, x = x[:, :n_prefix], x[:, n_prefix:]
        else:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)            # (B, HW, C)
        kv = self.kv_proj(text.to(x.dtype))             # (B, T, C)
        out, _ = self.attn(self.norm(x), kv, kv, key_padding_mask=pad_mask, need_weights=False)
        x = x + self.gate * out
        if tokens:
            return torch.cat([prefix, x], dim=1)
        return x.transpose(1, 2).reshape(B, C, H, W)


class ProgressiveTextFusion(nn.Module):
    """Progressive text context fusion on the FPN levels (paper Eq. 4) with Hi-SAM injection:
        x5 = Inject(p5, t_loc),            f_loc = U1(x5)
        x4 = Inject(p4 + f_loc, t_sty),    f_sty = U2(x4)
        x3 = Inject(p3 + f_sty, t_sem),    f_sem = U3(x3)
        p2' = p2 + f_sem
    U_j: 2x bilinear upsampling -> 3x3 conv/GN/ReLU -> 1x1 conv (zero-init, so the pyramid
    is unchanged at initialization). p3-p5 are replaced by x3-x5; other levels pass through."""

    LEVELS = (("p5", "location"), ("p4", "style"), ("p3", "semantic"))

    def __init__(self, dim=256, text_dim=512, num_heads=8, gate_init=0.05):
        super().__init__()
        self.text_adapters = nn.ModuleList(
            TextInjection(dim, text_dim, num_heads, gate_init) for _ in self.LEVELS
        )
        self.up = nn.ModuleList()
        for _ in self.LEVELS:
            proj = nn.Conv2d(dim, dim, kernel_size=1)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.up.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, dim),
                nn.ReLU(inplace=True),
                proj,
            ))

    def forward(self, features, text_ctx):
        out = dict(features)
        f = None
        for i, (name, role) in enumerate(self.LEVELS):
            x = features[name] if f is None else features[name] + f
            ctx = text_ctx.get(role)
            if ctx is not None:
                x = self.text_adapters[i](x, *ctx)
            out[name] = x
            f = self.up[i](x)
        out["p2"] = features["p2"] + f
        return out


def stride16_blocks(backbone):
    """(blocks of the stride-16 stage, their channel dim) of the backbone."""
    if hasattr(backbone, "net"):                       # DINOv3 ViT + SimpleFeaturePyramid
        return backbone.net.blocks, backbone.net.embed_dim
    bottom_up = backbone.bottom_up
    dim = bottom_up.output_shape()["res4"].channels
    if hasattr(bottom_up, "res4"):                     # detectron2 ResNet
        return bottom_up.res4, dim
    if hasattr(bottom_up, "model") and hasattr(bottom_up.model, "encoder"):  # ConvNeXt V2 (HF)
        return bottom_up.model.encoder.stages[2].layers, dim
    if hasattr(bottom_up, "stages") and hasattr(bottom_up, "downsample_layers"):  # DINOv3 ConvNeXt
        return bottom_up.stages[2], dim
    raise ValueError(f"Unsupported backbone for text injection: {type(bottom_up).__name__}")


# ----------------------------------------------------------------------------
# Glyph distance field (GDF)
# ----------------------------------------------------------------------------
class SineLayer(nn.Module):
    """SIREN layer: sin(w0 * (Wx + b)), with the initialization of Sitzmann et al. 2020.
    Implemented as a 1x1 conv so it runs as a per-pixel MLP on [B,C,H,W] maps."""

    def __init__(self, in_features, out_features, is_first=False, w0=30.0):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Conv2d(in_features, out_features, kernel_size=1)
        with torch.no_grad():
            bound = 1 / in_features if is_first else np.sqrt(6 / in_features) / w0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


class GDFPredictor(nn.Module):
    """Predicts the glyph distance field (GDF; >0 inside text, <0 outside, see gdf_targets) with a SIREN.

    A plain SIREN fits one signal from coordinates only; to predict a per-image GDF each
    pixel's input is [normalized p2 feature, (x, y) coordinate], i.e. a locally conditioned
    SIREN evaluated on the p2 grid."""

    def __init__(self, in_channels=256, feat_dim=64, hidden_dim=256, num_layers=3, w0=30.0):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, kernel_size=1),
            nn.GroupNorm(1, feat_dim),  # keep sine inputs O(1)
        )
        layers = [SineLayer(feat_dim + 2, hidden_dim, is_first=True, w0=w0)]
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, w0=w0))
        self.net = nn.Sequential(*layers)
        self.out = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden_dim) / w0
            self.out.weight.uniform_(-bound, bound)

    def forward(self, feat):
        B, _, H, W = feat.shape
        ys = torch.linspace(-1, 1, H, device=feat.device, dtype=feat.dtype)
        xs = torch.linspace(-1, 1, W, device=feat.device, dtype=feat.dtype)
        gy, gx = torch.meshgrid(ys, xs)
        coords = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        x = torch.cat([self.feat_proj(feat), coords], dim=1)
        return self.out(self.net(x))  # [B,1,H,W]


def gdf_targets(targets, size, ignore_value):
    """GT GDF (paper Eq. 2) at `size` from full-res labels (1=text, 0=bg, ignore_value=pad):
    0 on the text boundary, +d inside text, -d outside, where d is the Euclidean distance
    to the boundary in normalized image coordinates [-1, 1]^2.
    Returns (gdf [B,1,h,w], valid [B,1,h,w])."""
    from scipy.ndimage import binary_erosion, distance_transform_edt

    t = F.interpolate(targets.unsqueeze(1).float(), size=size, mode="nearest")
    valid = t != ignore_value
    text = (t == 1).squeeze(1).cpu().numpy()
    h, w = size
    spacing = (2.0 / h, 2.0 / w)  # pixel size in normalized coordinates
    gdfs = []
    for m in text:
        boundary = m & ~binary_erosion(m, border_value=1)
        if boundary.any():
            d = distance_transform_edt(~boundary, sampling=spacing)
            gdf = np.where(m, d, -d)
        else:  # no text: everything is far outside
            gdf = np.full(m.shape, 2.0 if m.all() else -2.0)
        gdfs.append(gdf)
    gdf = torch.from_numpy(np.stack(gdfs)).float().unsqueeze(1).to(targets.device)
    return gdf, valid


class GDFCrossAttention(nn.Module):
    """p2 pixels (query) attend to pooled GDF-map tokens. gate=0 init -> starts as a no-op."""

    def __init__(self, dim=256, num_tokens=256, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.hd = dim // heads
        self.grid = int(round(num_tokens ** 0.5))
        self.gdf_embed = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(self.grid)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, p2, gdf):
        B, C, H, W = p2.shape
        et = self.pool(self.gdf_embed(gdf)).flatten(2).transpose(1, 2)  # [B,T,C]
        q = self.q_proj(self.q_norm(p2.flatten(2).transpose(1, 2)))      # [B,N,C]
        k = self.k_proj(et)
        v = self.v_proj(et)

        def split(t):
            return t.view(t.size(0), t.size(1), self.heads, self.hd).transpose(1, 2)

        qh, kh, vh = split(q), split(k), split(v)
        attn = ((qh @ kh.transpose(-1, -2)) / (self.hd ** 0.5)).softmax(dim=-1)
        out = (attn @ vh).transpose(1, 2).reshape(B, H * W, C)
        out = self.o_proj(out).transpose(1, 2).reshape(B, C, H, W)
        return p2 + self.gate * out


# ----------------------------------------------------------------------------
# Losses for the coarse head
# ----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        mask = targets != self.ignore_index
        targets = targets[mask]
        logits_at_class1 = logits[:, 1, :, :][mask]
        ce_loss = F.binary_cross_entropy_with_logits(logits_at_class1, targets.float(), reduction="none")
        probs_at_class1 = torch.sigmoid(logits_at_class1)
        pt = torch.where(targets == 1, probs_at_class1, 1 - probs_at_class1)
        loss = self.alpha * (1 - pt).pow(self.gamma) * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1, ignore_value=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_value = ignore_value

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)[:, 1, :, :]
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)
        mask = (targets != self.ignore_value).float()
        probs = probs * mask
        targets = targets * mask
        intersection = (probs * targets).sum(1)
        dice_coeff = (2.0 * intersection + self.smooth) / (probs.sum(1) + targets.sum(1) + self.smooth)
        return 1 - dice_coeff.mean()


# ----------------------------------------------------------------------------
# Meta architecture
# ----------------------------------------------------------------------------
@META_ARCH_REGISTRY.register()
class TextSemanticSegmentor(nn.Module):
    @configurable
    def __init__(self, *, backbone: Backbone, sem_seg_head: nn.Module, pixel_mean, pixel_std,
                 inject_location: str = "backbone", inject_blocks: int = 3, inject_heads: int = 8,
                 inject_gate_init: float = 0.05, inject_block_ids=(), pv_cfg=None, role_cfg=None):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

        # CLIP ViT-B/32 text encoder: frozen + LoRA (r=8)
        clip_model, _ = clip.load("ViT-B/32", device="cuda")
        self.text_model = clip_model
        for param in self.text_model.parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        self.text_model.transformer = get_peft_model(self.text_model.transformer, lora_config)

        self._text_ctx = {}  # role -> (tokens, pad_mask), set for the duration of a forward
        self.inject_location = inject_location
        if inject_location == "fpn":
            dim = backbone.output_shape()["p2"].channels
            self.text_fusion = ProgressiveTextFusion(dim, 512, inject_heads, inject_gate_init)
        elif inject_location == "backbone":
            # Hi-SAM-style injection into the last `inject_blocks` blocks of the stride-16 stage
            blocks, dim = stride16_blocks(backbone)
            self.text_adapters = nn.ModuleList(
                TextInjection(dim, 512, inject_heads, inject_gate_init) for _ in range(inject_blocks)
            )
            n = len(blocks)
            if inject_block_ids:
                # Hi-SAM --pv_blocks l,s,m: one block per cue, location first
                ids = [int(v) for v in inject_block_ids]
                assert len(ids) == 3 == inject_blocks and ids == sorted(ids), ids
                plan = list(zip(ids, ("location", "style", "semantic")))
            else:
                plan = [(n - inject_blocks + i, TEXT_ORDER[i] if inject_blocks == 3 else TEXT_ORDER[min(i, 2)])
                        for i in range(inject_blocks)]
            for i, (b, role) in enumerate(plan):
                blocks[b].register_forward_hook(self._make_hook(i, role))
                blocks[b].has_text_hook = True
        else:
            raise ValueError(f"Unknown MODEL.TEXT_INJECTION.LOCATION: {inject_location}")

        # Hi-SAM role heads read the (text-injected) output of each cue's block
        self.role_heads = None
        self._role_feats = {}
        self.aux_losses_off = False  # set by the trainer around the text-consistency forwards
        if role_cfg is not None and role_cfg.ENABLED:
            assert inject_location == "backbone" and inject_block_ids, \
                "role heads need backbone injection with explicit TEXT_INJECTION.BLOCKS"
            self.role_heads = RoleHeads(dim, role_cfg.REGION_WEIGHT, role_cfg.SDF_WEIGHT, role_cfg.SKEL_WEIGHT)

        self.pv = None
        if pv_cfg is not None and pv_cfg.ENABLED:
            self.pv = PVHeads(dim=sem_seg_head.coarse_sem_seg_head.gdf_predictor.feat_proj[0].in_channels,
                              text_dim=512, ignore_value=255,
                              w_prop=pv_cfg.PROPOSAL_WEIGHT, w_sdf=pv_cfg.WORD_SDF_WEIGHT,
                              w_skel=pv_cfg.SKEL_WEIGHT, w_ver=pv_cfg.VERIFY_WEIGHT)
            self.pv_max_words = pv_cfg.MAX_WORDS
            self.pv_stats = {}

    def _make_hook(self, idx, role):
        def hook(module, inputs, output):
            ctx = self._text_ctx.get(role)
            if ctx is None:  # no text: visual path only
                return output
            n_prefix = getattr(module, "n_prefix_tokens", 0)
            out = self.text_adapters[idx](output, *ctx, n_prefix=n_prefix)
            if self.role_heads is not None and self.training and not self.aux_losses_off:
                if out.dim() == 4:  # conv stage: already (B, C, h, w)
                    self._role_feats[role] = out
                else:               # ViT tokens
                    H, W = self.backbone.net.grid
                    self._role_feats[role] = out[:, n_prefix:].transpose(1, 2).reshape(out.shape[0], -1, H, W)
            return out
        return hook

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        t = cfg.MODEL.TEXT_INJECTION
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "inject_location": t.LOCATION,
            "inject_blocks": t.NUM_BLOCKS,
            "inject_heads": t.NUM_HEADS,
            "inject_gate_init": t.GATE_INIT,
            "inject_block_ids": t.BLOCKS,
            "pv_cfg": cfg.MODEL.PV,
            "role_cfg": cfg.MODEL.ROLE_HEADS,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _encode_text_pertoken(self, tokens):
        """CLIP text encoder -> per-token features [B, T', 512] (no pooling, PAD zeroed,
        T' cropped to the longest non-PAD length in the batch)."""
        m = self.text_model
        x = m.token_embedding(tokens).type(m.dtype)
        x = x + m.positional_embedding.type(m.dtype)
        x = m.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = m.ln_final(x).type(m.dtype)
        if getattr(m, "text_projection", None) is not None:
            x = x @ m.text_projection
        pad = tokens.eq(0)
        x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        keep = (~pad).any(dim=0)
        last = int(keep.nonzero().max().item()) + 1 if keep.any() else 1
        return x[:, :last].float()

    def _encode_prompts(self, batched_inputs):
        if "location_prompt_tokens" in batched_inputs[0]:
            # pre-tokenized (DefaultPredictor path)
            toks = [torch.stack([x[k].to(self.device) for x in batched_inputs])
                    for k in ("location_prompt_tokens", "style_prompt_tokens", "recognition_prompt_tokens")]
        elif "location_text" in batched_inputs[0]:
            # raw strings (TextPromptDatasetMapper path)
            toks = [clip.tokenize([x.get(k, "") for x in batched_inputs], truncate=True).to(self.device)
                    for k in ("location_text", "style_text", "recognition_text")]
        else:
            return None, None, None
        return tuple(self._encode_text_pertoken(t) for t in toks)

    def _pv_losses(self, batched_inputs, images):
        """PV structure heads + verifier on the coarse head's fused stride-4 feature (see pv_heads.py)."""
        head = self.sem_seg_head.coarse_sem_seg_head
        feat = head.last_feat
        targets = ImageList.from_tensors([x["sem_seg"].to(self.device) for x in batched_inputs],
                                         self.backbone.size_divisibility, 255).tensor
        word_maps = None
        if "word_map" in batched_inputs[0]:
            word_maps = ImageList.from_tensors([x["word_map"].to(self.device) for x in batched_inputs],
                                               self.backbone.size_divisibility, 0).tensor
        skel = None
        if "skel_heat" in batched_inputs[0]:
            h, w = feat.shape[-2:]
            skel = torch.stack([F.pad(x["skel_heat"].to(self.device),
                                      (0, w - x["skel_heat"].shape[-1], 0, h - x["skel_heat"].shape[-2]))
                                for x in batched_inputs])[:, None]
        out = self.pv.dense_losses(feat, targets, word_maps, skel)
        if word_maps is not None and "recognition_text" in batched_inputs[0]:
            words = [split_semantic_words(x["recognition_text"])[:self.pv_max_words] for x in batched_inputs]
            flat = [w for ws in words for w in ws]
            emb = []
            if flat:
                with torch.no_grad():   # fixed word features (Hi-SAM: precomputed CLIP), verifier-side projection learns
                    tok = clip.tokenize(flat, truncate=True).to(self.device)
                    per = self._encode_text_pertoken(tok)
                    eot = tok.argmax(-1).clamp_max(per.shape[1] - 1)
                    e = per[torch.arange(len(flat), device=self.device), eot]
                i = 0
                for ws in words:
                    emb.append(e[i:i + len(ws)]); i += len(ws)
            else:
                emb = [None] * len(words)
            out.update(self.pv.verify_loss(feat, word_maps, [x.get("word_texts", []) for x in batched_inputs],
                                           words, emb, self.pv_stats))
        return out

    def _role_losses(self, batched_inputs, targets):
        word_maps = skel = None
        if "word_map" in batched_inputs[0]:
            word_maps = ImageList.from_tensors([x["word_map"].to(self.device) for x in batched_inputs],
                                               self.backbone.size_divisibility, 0).tensor
        if "skel_heat" in batched_inputs[0]:
            h, w = max(x["skel_heat"].shape[-2] for x in batched_inputs), max(x["skel_heat"].shape[-1] for x in batched_inputs)
            skel = torch.stack([F.pad(x["skel_heat"].to(self.device),
                                      (0, w - x["skel_heat"].shape[-1], 0, h - x["skel_heat"].shape[-2]))
                                for x in batched_inputs])[:, None]
        return self.role_heads.losses(self._role_feats, targets, word_maps, skel)

    def forward(self, batched_inputs):
        self._role_feats = {}
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images, self.backbone.size_divisibility, padding_constraints=self.backbone.padding_constraints
        )

        location, style, recognition = self._encode_prompts(batched_inputs)
        self._text_ctx = {}
        for role, t in (("semantic", recognition), ("style", style), ("location", location)):
            if t is not None:
                pad = t.abs().sum(-1) == 0
                pad = pad & ~pad.all(dim=1, keepdim=True)  # all-PAD row -> unmask (avoid NaN)
                self._text_ctx[role] = (t, pad)
        try:
            features = self.backbone(images.tensor)
            if self.inject_location == "fpn":
                features = self.text_fusion(features, self._text_ctx)
        finally:
            self._text_ctx = {}

        targets = None
        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor

        results, losses = self.sem_seg_head(features, targets)
        if self.training:
            if self.pv is not None and not self.aux_losses_off:
                losses.update(self._pv_losses(batched_inputs, images))
            if self._role_feats:
                losses.update(self._role_losses(batched_inputs, targets))
            self._role_feats = {}
            return losses
        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({"sem_seg": sem_seg_postprocess(result, image_size, height, width)})
        return processed_results


# ----------------------------------------------------------------------------
# Coarse head: FPN + GDF
# ----------------------------------------------------------------------------
@SEM_SEG_HEADS_REGISTRY.register()
class TextSemSegFPNHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        conv_dims: int,
        common_stride: int,
        loss_weight: float = 1.0,
        norm: Optional[Union[str, Callable]] = None,
        ignore_value: int = -1,
        gdf_loss_weight: float = 1.0,
        gdf_hidden_dim: int = 256,
        gdf_num_layers: int = 3,
        gdf_w0: float = 30.0,
    ):
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        if not len(input_shape):
            raise ValueError("TextSemSegFPNHead(input_shape=) cannot be empty!")
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = common_stride
        self.loss_weight = loss_weight
        self.focal_loss = FocalLoss()
        self.per_image_q = False   # set by the trainer around the text-consistency forwards
        self.dice_loss = DiceLoss()

        # PanopticFPN scale heads: every level -> stride `common_stride`, then summed
        self.scale_heads = []
        for in_feature, stride, channels in zip(self.in_features, feature_strides, feature_channels):
            head_ops = []
            head_length = max(1, int(np.log2(stride) - np.log2(self.common_stride)))
            for k in range(head_length):
                conv = Conv2d(
                    channels if k == 0 else conv_dims, conv_dims, kernel_size=3, stride=1, padding=1,
                    bias=not norm, norm=get_norm(norm, conv_dims), activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if stride != self.common_stride:
                    head_ops.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])

        self.gdf_predictor = GDFPredictor(
            in_channels=conv_dims, hidden_dim=gdf_hidden_dim, num_layers=gdf_num_layers, w0=gdf_w0
        )
        self.gdf_loss_weight = gdf_loss_weight
        self.gdf_cross_attention = GDFCrossAttention(dim=conv_dims, num_tokens=256, heads=8)

        # coarse segmentor: 2x (conv3x3 -> ReLU -> BN) -> 1x1
        self.coarse_layer = nn.Conv2d(conv_dims, 256, kernel_size=3, padding=1)
        self.coarse_layer2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.predictor = Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "conv_dims": cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            "common_stride": cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            "norm": cfg.MODEL.SEM_SEG_HEAD.NORM,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "gdf_loss_weight": cfg.MODEL.GDF.LOSS_WEIGHT,
            "gdf_hidden_dim": cfg.MODEL.GDF.HIDDEN_DIM,
            "gdf_num_layers": cfg.MODEL.GDF.NUM_LAYERS,
            "gdf_w0": cfg.MODEL.GDF.W0,
        }

    def forward(self, features, targets=None):
        x, gdf = self.layers(features)
        if self.training:
            return None, self.losses(x, targets, gdf)
        x = F.interpolate(x, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        return x, {}

    def layers(self, features):
        """Returns (coarse logits [B,C,H/4,W/4], predicted GDF [B,1,H/4,W/4])."""
        x = None
        for i, f in enumerate(self.in_features):
            y = self.scale_heads[i](features[f])
            x = y if x is None else x + y
        self.last_feat = x   # fused stride-4 feature, read by the PV heads (MODEL.PV)
        gdf = self.gdf_predictor(x)
        x = self.gdf_cross_attention(x, gdf.clamp(-1, 1))
        x = self.batch_norm1(F.relu(self.coarse_layer(x)))
        x = self.batch_norm2(F.relu(self.coarse_layer2(x)))
        return self.predictor(x), gdf

    def per_image_seg_loss(self, predictions, targets):
        """loss_sem_seg (dice + focal + ce) computed per image -> (B,). Used by the content hinge."""
        valid = (targets != self.ignore_value).float()
        t = targets.clamp_max(1)
        n = valid.flatten(1).sum(1).clamp_min(1)
        ce = (F.cross_entropy(predictions, targets, reduction="none", ignore_index=self.ignore_value)
              * valid).flatten(1).sum(1) / n
        p1 = torch.sigmoid(predictions[:, 1])
        bce = F.binary_cross_entropy_with_logits(predictions[:, 1], t.float(), reduction="none")
        pt = torch.where(t == 1, p1, 1 - p1)
        a = self.focal_loss
        focal = (a.alpha * (1 - pt).pow(a.gamma) * bce * valid).flatten(1).sum(1) / n
        probs = F.softmax(predictions, dim=1)[:, 1] * valid
        tf = t.float() * valid
        inter = (probs * tf).flatten(1).sum(1)
        dice = 1 - (2.0 * inter + 1) / (probs.flatten(1).sum(1) + tf.flatten(1).sum(1) + 1)
        return dice + focal + ce

    def losses(self, predictions, targets, gdf_pred=None):
        predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=True
        )
        cs_loss = F.cross_entropy(predictions, targets, reduction="mean", ignore_index=self.ignore_value)
        focal_loss = self.focal_loss(predictions, targets)
        dice_loss = self.dice_loss(predictions, targets)
        losses = {"loss_sem_seg": dice_loss + focal_loss + cs_loss}
        if self.per_image_q:
            losses["q_img"] = self.per_image_seg_loss(predictions, targets)
        if gdf_pred is not None:
            gdf_gt, valid = gdf_targets(targets, gdf_pred.shape[-2:], self.ignore_value)
            l1 = (gdf_pred.float() - gdf_gt).abs() * valid
            losses["loss_gdf"] = self.gdf_loss_weight * l1.sum() / valid.sum().clamp_min(1)
        return losses
