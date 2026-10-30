# Copyright (c) Facebook, Inc. and its affiliates.
"""
LaTeX-Seg segmentation head:
    coarse FPN head on text-conditioned features (see text_fpn.py) -> implicit glyph refinement (IGR).
"""
from typing import Dict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .point_features import get_uncertain_point_coords_on_grid, point_sample
from .implicit_glyph_refinement import build_igr


def sample_gdf_band_points(gdf, size, band, num_points):
    """
    Uniformly sample `num_points` points per image from the boundary band
    X_b = {p : |GDF(p)| < band} (paper Eq. 7).

    Args:
        gdf (Tensor): (N, 1, h, w) predicted GDF, evaluated at full resolution `size`.
        size: (H, W) image size.
    Returns:
        (N, P, 2) (x, y) point coordinates in [0, 1]. An image whose band is empty
        (e.g. no text) gets points uniformly over the whole image.
    """
    H, W = size
    device = gdf.device
    in_band = F.interpolate(gdf, size=size, mode="bilinear", align_corners=False)[:, 0].abs() < band
    coords = []
    for m in in_band:
        idx = m.flatten().nonzero()[:, 0]
        if idx.numel() == 0:
            idx = torch.arange(H * W, device=device)
        if idx.numel() >= num_points:
            sel = idx[torch.randperm(idx.numel(), device=device)[:num_points]]
        else:
            sel = idx[torch.randint(idx.numel(), (num_points,), device=device)]
        jitter = torch.rand(num_points, 2, device=device)  # continuous position inside the pixel
        x = ((sel % W).float() + jitter[:, 0]) / W
        y = ((sel // W).float() + jitter[:, 1]) / H
        coords.append(torch.stack([x, y], dim=-1))
    return torch.stack(coords)

def calculate_uncertainty(sem_seg_logits):
    """
    Uncertainty = difference between the top-2 logits (higher = more uncertain).
    Args:
        sem_seg_logits (Tensor): (N, C, ...) logits.
    Returns:
        (N, 1, ...) uncertainty scores.
    """
    top2_scores = torch.topk(sem_seg_logits, k=2, dim=1)[0]
    return (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)


@SEM_SEG_HEADS_REGISTRY.register()
class LaTeXSegHead(nn.Module):
    """
    Combines the coarse head set in `MODEL.SEM_SEG_HEAD.COARSE_HEAD_NAME`
    and the IGR module set in `MODEL.IGR.NAME`.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.coarse_sem_seg_head = SEM_SEG_HEADS_REGISTRY.get(
            cfg.MODEL.SEM_SEG_HEAD.COARSE_HEAD_NAME
        )(cfg, input_shape)
        self.igr_enabled = cfg.MODEL.IGR.ENABLED
        assert not (self.igr_enabled and cfg.MODEL.RGB_SKIP.ENABLED), "RGB_SKIP is for the IGR-off path"
        if self.igr_enabled:
            self._init_igr(cfg, input_shape)

    def _init_igr(self, cfg, input_shape: Dict[str, ShapeSpec]):
        assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == cfg.MODEL.IGR.NUM_CLASSES
        feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.in_features = cfg.MODEL.IGR.IN_FEATURES
        self.gdf_band = cfg.MODEL.IGR.GDF_BAND
        self.subdivision_steps = cfg.MODEL.IGR.SUBDIVISION_STEPS
        self.train_num_points = cfg.MODEL.IGR.TRAIN_NUM_POINTS
        self.subdivision_num_points = cfg.MODEL.IGR.SUBDIVISION_NUM_POINTS
        in_channels = int(np.sum([feature_channels[f] for f in self.in_features]))
        self.igr = build_igr(cfg, ShapeSpec(channels=in_channels, width=1, height=1))

    def forward(self, features, targets=None):
        coarse_sem_seg_logits, gdf_pred = self.coarse_sem_seg_head.layers(features)

        if self.training:
            losses = self.coarse_sem_seg_head.losses(coarse_sem_seg_logits, targets, gdf_pred)
            if not self.igr_enabled:
                return None, losses

            with torch.no_grad():
                point_coords = sample_gdf_band_points(
                    gdf_pred.float(), targets.shape[-2:], self.gdf_band, self.train_num_points
                )
            coarse_features = point_sample(coarse_sem_seg_logits, point_coords, align_corners=False)
            point_logits = self.igr(
                coarse_features, features, point_coords, coarse_sem_seg_logits
            )
            point_targets = (
                point_sample(
                    targets.unsqueeze(1).to(torch.float),
                    point_coords,
                    mode="nearest",
                    align_corners=False,
                )
                .squeeze(1)
                .to(torch.long)
            )
            losses["loss_igr"] = F.cross_entropy(
                point_logits, point_targets, reduction="mean", ignore_index=self.ignore_value
            )
            return None, losses

        if not self.igr_enabled:
            up = F.interpolate(coarse_sem_seg_logits, scale_factor=4, mode="bilinear", align_corners=False)
            return self.coarse_sem_seg_head.refine(up), {}
        sem_seg_logits = coarse_sem_seg_logits.clone()
        for _ in range(self.subdivision_steps):
            sem_seg_logits = F.interpolate(
                sem_seg_logits, scale_factor=2, mode="bilinear", align_corners=False
            )
            uncertainty_map = calculate_uncertainty(sem_seg_logits)
            num_points = min(self.subdivision_num_points, sem_seg_logits.size(-2) * sem_seg_logits.size(-1))
            point_indices, point_coords = get_uncertain_point_coords_on_grid(
                uncertainty_map, num_points
            )
            coarse_features = point_sample(coarse_sem_seg_logits, point_coords, align_corners=False)
            point_logits = self.igr(
                coarse_features, features, point_coords, coarse_sem_seg_logits
            )

            N, C, H, W = sem_seg_logits.shape
            point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
            sem_seg_logits = (
                sem_seg_logits.reshape(N, C, H * W)
                .scatter_(2, point_indices, point_logits)
                .view(N, C, H, W)
            )
        return sem_seg_logits, {}
