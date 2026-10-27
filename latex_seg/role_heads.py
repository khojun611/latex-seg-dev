"""Hi-SAM role heads on the text-injection blocks of a ViT (MODEL.ROLE_HEADS).

As in Hi-SAM (pv_blocks l,s,m with the heads on the injection blocks), each cue has its own block and the
output of that block (after the text injection) feeds a dense head:
  location block -> region head   : 'any text' in each stride-4 cell                      -> bce + dice
  style block    -> SDF head      : [glyph SDF (truncated +-8 px at stride 4), word SDF]  -> L1
  semantic block -> skeleton head : gaussian heat around the pruned medial axis           -> bce + dice
The heads read the stride-16 tokens and predict at stride 4 (Hi-SAM: 256x256 for a 1024 input).
"""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .pv_heads import bce_dice, word_sdf_from_map

ROLE_OUT = {"location": 1, "style": 2, "semantic": 1}


class RoleHead(nn.Module):
    """stride-16 tokens (B, C, h, w) -> stride-4 map (B, out, 4h, 4w)."""

    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 1), nn.GroupNorm(32, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(32, hidden), nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden // 2, 2, stride=2), nn.GroupNorm(32, hidden // 2), nn.GELU(),
            nn.ConvTranspose2d(hidden // 2, hidden // 4, 2, stride=2), nn.GroupNorm(16, hidden // 4), nn.GELU(),
            nn.Conv2d(hidden // 4, out_dim, 1),
        )

    def forward(self, x):
        return self.up(self.reduce(x))


def glyph_sdf(text4, trunc=8.0):
    """(h, w) binary text map at stride 4 -> signed distance, + inside / - outside, truncated to +-trunc px, /trunc."""
    t = text4.astype(np.uint8)
    if not t.any():
        return np.full(t.shape, -1.0, np.float32)
    d_in = cv2.distanceTransform(t, cv2.DIST_L2, 3)
    d_out = cv2.distanceTransform(1 - t, cv2.DIST_L2, 3)
    return (np.clip(d_in - d_out, -trunc, trunc) / trunc).astype(np.float32)


class RoleHeads(nn.Module):
    def __init__(self, in_dim, w_region=1.0, w_sdf=0.5, w_skel=0.5, ignore_value=255):
        super().__init__()
        self.heads = nn.ModuleDict({r: RoleHead(in_dim, n) for r, n in ROLE_OUT.items()})
        self.w = {"location": w_region, "style": w_sdf, "semantic": w_skel}
        self.ignore_value = ignore_value

    def losses(self, feats, targets, word_maps=None, skel_heat=None):
        """feats: role -> (B, C, h, w) block outputs; targets (B, H, W) labels; word_maps (B, H, W) ids;
        skel_heat (B, 1, H/4, W/4) from the mapper."""
        out = {}
        ref = next(iter(feats.values()))
        B = ref.shape[0]
        h, w = ref.shape[-2] * 4, ref.shape[-1] * 4
        text4 = F.adaptive_max_pool2d((targets == 1).float()[:, None], (h, w))
        valid4 = F.adaptive_max_pool2d((targets != self.ignore_value).float()[:, None], (h, w)) > 0
        if "location" in feats and self.w["location"] > 0:
            out["loss_role_region"] = self.w["location"] * bce_dice(self.heads["location"](feats["location"]),
                                                                    text4, valid4)
        if "style" in feats and self.w["style"] > 0:
            pred = self.heads["style"](feats["style"]).float()
            t4 = text4[:, 0].cpu().numpy() > 0.5
            shape = torch.from_numpy(np.stack([glyph_sdf(m) for m in t4]))[:, None].to(pred.device)
            vm = valid4.float()
            loss = ((pred[:, :1] - shape).abs() * vm).sum() / vm.sum().clamp_min(1)
            if word_maps is not None:
                wm = F.interpolate(word_maps[:, None].float(), size=(h, w), mode="nearest")[:, 0]
                sdfs, oks = zip(*[word_sdf_from_map(m) for m in wm.cpu().numpy().astype(np.uint8)])
                ok = torch.tensor(oks, device=pred.device, dtype=torch.float32)[:, None, None, None]
                if ok.any():
                    tgt = torch.from_numpy(np.stack(sdfs))[:, None].to(pred.device)
                    vw = vm * ok
                    loss = loss + ((pred[:, 1:] - tgt).abs() * vw).sum() / vw.sum().clamp_min(1)
            out["loss_role_sdf"] = self.w["style"] * loss
        if "semantic" in feats and self.w["semantic"] > 0 and skel_heat is not None:
            sk = F.pad(skel_heat, (0, w - skel_heat.shape[-1], 0, h - skel_heat.shape[-2]))[..., :h, :w]
            out["loss_role_skel"] = self.w["semantic"] * bce_dice(self.heads["semantic"](feats["semantic"]),
                                                                  sk.float(), valid4)
        return out
