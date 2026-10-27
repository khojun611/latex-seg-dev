# Copyright (c) Facebook, Inc. and its affiliates.
"""
Implicit Glyph Refinement (IGR): local implicit (LIIF/LTE-style) refinement of sampled points.

For each query point the 4 nearest p2 cells are visited. Each cell contributes
    coef(p2) * [cos, sin](pi * (freq(p2) . rel_coord + phase(cell)))
which an MLP maps to class logits; the 4 predictions are blended by bilinear
area weights and added to the coarse logits sampled at the point.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

IGR_REGISTRY = Registry("IGR")
IGR_REGISTRY.__doc__ = """
Registry for implicit glyph refinement modules, which predict logits at given points.
The registered object will be called with `obj(cfg, input_shape)`.
"""


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


def make_coord(shape, ranges=None, flatten=True):
    """Coordinates of cell centers for a grid of `shape`, in [-1, 1]."""
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = (-1, 1) if ranges is None else ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


@IGR_REGISTRY.register()
class ImplicitGlyphRefinement(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()
        num_classes = cfg.MODEL.IGR.NUM_CLASSES
        self.input_channels = input_shape.channels
        hidden_dim = 256
        self.coef = nn.Conv2d(self.input_channels, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.input_channels, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim // 2, bias=False)
        self.imnet = MLP(in_dim=hidden_dim, out_dim=num_classes, hidden_list=[256, 256, 256])

    def forward(self, coarse_features, features, point_coords, coarse_sem_seg_logits):
        """
        Args:
            coarse_features: (N, C, P) coarse logits sampled at the points.
            features: dict of FPN features; "p2" is used.
            point_coords: (N, P, 2) (x, y) in [0, 1] (point_features convention).
            coarse_sem_seg_logits: (N, C, H, W) coarse prediction (only its size is used).
        Returns:
            (N, C, P) point logits.
        """
        # LIIF convention below: (y, x) in [-1, 1]
        point_coords = point_coords.flip(-1) * 2 - 1
        feat = features["p2"]
        feat_shape = feat.shape
        feat_coord = (
            make_coord(feat_shape[-2:], flatten=False).to(feat.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(feat_shape[0], 2, *feat_shape[-2:])
        )
        coef = self.coef(feat)
        freq = self.freq(feat)

        cell = torch.ones_like(point_coords)
        cell[:, :, 0] *= 2 / coarse_sem_seg_logits.shape[-2]
        cell[:, :, 1] *= 2 / coarse_sem_seg_logits.shape[-1]

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = point_coords.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                grid = coord_.flip(-1).unsqueeze(1)
                q_coef = F.grid_sample(coef, grid, mode="nearest", align_corners=False)[
                    :, :, 0, :
                ].permute(0, 2, 1)
                q_freq = F.grid_sample(freq, grid, mode="nearest", align_corners=False)[
                    :, :, 0, :
                ].permute(0, 2, 1)
                q_coord = F.grid_sample(feat_coord, grid, mode="nearest", align_corners=False)[
                    :, :, 0, :
                ].permute(0, 2, 1)
                rel_coord = point_coords - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]

                bs, q = point_coords.shape[:2]
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                q_freq = torch.cat((torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1)

                x = torch.mul(q_coef, q_freq)
                pred = self.imnet(x.contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        # bilinear blending: each prediction is weighted by the area of the diagonal cell
        tot_area = torch.stack(areas).sum(dim=0)
        areas[0], areas[3] = areas[3], areas[0]
        areas[1], areas[2] = areas[2], areas[1]
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        ret = ret + coarse_features.permute(0, 2, 1)
        return ret.permute(0, 2, 1)


def build_igr(cfg, input_channels):
    """
    Build the IGR module defined by `cfg.MODEL.IGR.NAME`.
    """
    name = cfg.MODEL.IGR.NAME
    return IGR_REGISTRY.get(name)(cfg, input_channels)
