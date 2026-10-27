"""Glyph skeleton target: medial axis -> spur pruning -> gaussian heatmap.

A branch (endpoint .. junction) is a spur when its disks barely leave the junction's disk:
protrusion = max_p(|p - j| + r(p)) - r(j) < spur_ratio * r(j). Boundary-noise / corner spurs reach
~0.4 r(j); real short strokes (the middle arm of an E) reach well beyond it.
"""
import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import medial_axis

_K = np.ones((3, 3)); _K[1, 1] = 0


def _branch(sk, deg, e):
    path, prev, cur = [e], None, e
    while True:
        y, x = cur
        nb = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
              if (dy or dx) and 0 <= y + dy < sk.shape[0] and 0 <= x + dx < sk.shape[1]
              and sk[y + dy, x + dx] and (y + dy, x + dx) != prev and (y + dy, x + dx) not in path]
        if len(nb) != 1:
            return path, None
        prev, cur = cur, nb[0]
        if deg[cur] >= 3:
            return path, cur
        path.append(cur)


def prune_skeleton(sk, dist, spur_ratio=0.8, iters=4, min_cc=10):
    sk = sk.copy()
    for _ in range(iters):
        deg = ndimage.convolve(sk.astype(int), _K, mode="constant") * sk
        removed = 0
        for e in zip(*np.nonzero(deg == 1)):
            if not sk[e]:
                continue
            path, j = _branch(sk, deg, e)
            if j is None:
                continue
            p = np.asarray(path, float)
            reach = np.hypot(p[:, 0] - j[0], p[:, 1] - j[1]) + dist[tuple(np.asarray(path).T)]
            if reach.max() - dist[j] < spur_ratio * max(dist[j], 1.0):
                for q in path:
                    sk[q] = False
                removed += 1
        if not removed:
            break
    lab, n = ndimage.label(sk, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(sk, lab, range(1, n + 1))
        sk &= np.isin(lab, 1 + np.nonzero(sizes >= min_cc)[0])
    return sk


def skeleton_heatmap(mask, out_hw=(256, 256), sigma=1.5, spur_ratio=0.8):
    """Text mask (H,W) -> (pruned full-res skeleton, gaussian heat in [0,1] at out_hw)."""
    sk, dist = medial_axis(mask > 0, return_distance=True)
    sk = prune_skeleton(sk, dist, spur_ratio)
    H, W = mask.shape[:2]
    r = np.zeros(out_hw, np.uint8)
    pts = np.argwhere(sk)
    if len(pts):
        ys = np.clip(np.round((pts[:, 0] + 0.5) * out_hw[0] / H - 0.5).astype(int), 0, out_hw[0] - 1)
        xs = np.clip(np.round((pts[:, 1] + 0.5) * out_hw[1] / W - 0.5).astype(int), 0, out_hw[1] - 1)
        r[ys, xs] = 1
    d = cv2.distanceTransform(1 - r, cv2.DIST_L2, 3)
    return sk, np.exp(-d ** 2 / (2 * sigma ** 2)).astype(np.float32) * (r.max() > 0)
