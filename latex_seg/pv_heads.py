"""Hi-SAM propose-and-verify (PV) structure heads, ported to LaTeX-Seg (MODEL.PV.ENABLED).

On the fused stride-4 feature of the coarse head (the map the GDF predictor reads):
  proposal : text-region map (target: 'any text' in each stride-4 cell)            -> bce + dice
  word SDF : signed distance to each word's own boundary (TextSeg word quads)       -> L1
  skeleton : gaussian heat around the pruned medial axis of the text mask           -> bce + dice
  verify   : GT word regions (pooled feature) x prompt words (CLIP EOT of each word)-> balanced BCE
The glyph SDF of Hi-SAM is LaTeX-Seg's own GDF head. Targets follow Hi-SAM's dataloader / train.py.
"""
import difflib
import json
import re

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .skeleton_target import skeleton_heatmap

# ----------------------------------------------------------------------------
# words (as Hi-SAM hi_sam/data/dataloader.py)
# ----------------------------------------------------------------------------
_PV_LEAD = re.compile(r"^\s*the\s+text\s+(with|reading|located\s+at)\s+", re.I)
_PV_SKIP = re.compile(r"no legible|no readable|illegible|unreadable|no (visible )?text|don't know", re.I)


def split_semantic_words(text):
    """"The text reading 'WHY', 'ME', and '?'" -> ['WHY', 'ME', '?'] (illegible entries dropped)."""
    s = _PV_LEAD.sub("", text or "").strip()
    words = re.findall(r"'(.+?)'(?=\s*(?:,|;|\band\b|\.|$))", s)
    if not words:
        words = [w.strip().strip("'\".") for w in re.split(r",|\band\b", s)]
    out = []
    for w in words:
        w = w.strip()
        if w and not _PV_SKIP.search(w) and w not in out:
            out.append(w)
    return out


def _norm_word(w):
    n = re.sub(r"[^0-9a-z]", "", w.casefold())
    return n if n else w.strip()


def match_words(gt_word, prompt_word):
    """GT word (TextSeg anno) <-> prompt word/phrase (GPT). Exact, token-of-phrase, or fuzzy."""
    g, p = _norm_word(gt_word), _norm_word(prompt_word)
    if not g or not p:
        return False
    if g == p or g in [_norm_word(t) for t in prompt_word.split() if t]:
        return True
    return len(g) >= 3 and difflib.SequenceMatcher(None, g, p).ratio() >= 0.8


def load_word_quads(anno_path):
    """TextSeg <stem>_anno.json -> [(text, (4,2) int32 quad)] in key order."""
    with open(anno_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for k in sorted(d.keys()):
        bb = d[k].get("bbox", [])
        if isinstance(bb, str):
            bb = json.loads(bb)
        if len(bb) >= 8:
            out.append((str(d[k].get("text", "")), np.asarray(bb[:8], np.int32).reshape(4, 2)))
    return out


def word_map_from_quads(quads, h, w, max_words=255):
    wm = np.zeros((h, w), np.uint8)
    for wi, (_, q) in enumerate(quads[:max_words], 1):
        cv2.fillPoly(wm, [q], wi)
    return wm


# ----------------------------------------------------------------------------
# targets (as Hi-SAM)
# ----------------------------------------------------------------------------
def word_sdf_from_map(wm):
    """Word instance map (h,w ids) -> signed distance in [-1,1] (1 at a word centre, 0 at its edge,
    -distance / median word radius outside)."""
    sdf = np.full(wm.shape, -1.0, np.float32)
    ids = [i for i in np.unique(wm) if i > 0]
    if not ids:
        return sdf, False
    radii = []
    for i in ids:
        m = (wm == i).astype(np.uint8)
        d_in = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        r = float(d_in.max())
        if r <= 0:
            continue
        radii.append(r)
        sdf[m > 0] = d_in[m > 0] / r
    scale = float(np.median(radii)) if radii else 1.0
    d_out = cv2.distanceTransform((wm == 0).astype(np.uint8), cv2.DIST_L2, 3)
    out = wm == 0
    sdf[out] = -np.minimum(d_out[out] / max(scale, 1.0), 1.0)
    return sdf, bool(radii)


def bce_dice(logit, target, valid=None):
    logit, target = logit.float(), target.float()
    w = torch.ones_like(target) if valid is None else valid.float()
    bce = (F.binary_cross_entropy_with_logits(logit, target, reduction="none") * w).sum() / w.sum().clamp_min(1)
    p = (torch.sigmoid(logit) * w).flatten(1)
    t = (target * w).flatten(1)
    dice = 1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)
    return bce + dice.mean()


# ----------------------------------------------------------------------------
# heads
# ----------------------------------------------------------------------------
def _dense_head(dim, hidden=128):
    return nn.Sequential(
        nn.Conv2d(dim, hidden, 3, padding=1), nn.GroupNorm(16, hidden), nn.ReLU(inplace=True),
        nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(16, hidden), nn.ReLU(inplace=True),
        nn.Conv2d(hidden, 1, 1),
    )


class PVHeads(nn.Module):
    def __init__(self, dim=256, text_dim=512, embed_dim=256, ignore_value=255,
                 w_prop=1.0, w_sdf=0.5, w_skel=0.5, w_ver=0.5):
        super().__init__()
        self.prop_head = _dense_head(dim)
        self.sdf_head = _dense_head(dim)
        self.skel_head = _dense_head(dim)
        # verifier (Hi-SAM CueVerifier): pooled region feature / word embedding -> cosine * scale + bias
        self.ver_vis = nn.Sequential(nn.Linear(dim, embed_dim), nn.ReLU(inplace=True), nn.Linear(embed_dim, embed_dim))
        self.ver_txt = nn.Linear(text_dim, embed_dim)
        self.ver_scale = nn.Parameter(torch.tensor(10.0))
        self.ver_bias = nn.Parameter(torch.tensor(-5.0))
        self.ignore_value = ignore_value
        self.w = dict(prop=w_prop, sdf=w_sdf, skel=w_skel, ver=w_ver)

    def dense_losses(self, feat, targets, word_maps, skel_heat=None):
        """feat (B,C,h,w) stride-4; targets (B,H,W) labels with ignore; word_maps (B,H,W) ids or None;
        skel_heat (B,1,h,w) from the mapper (computed here if None)."""
        B, _, h, w = feat.shape
        out = {}
        valid_full = (targets != self.ignore_value).float()
        text_full = ((targets == 1).float())
        text4 = F.adaptive_max_pool2d(text_full[:, None], (h, w))
        valid4 = F.adaptive_max_pool2d(valid_full[:, None], (h, w)) > 0
        if self.w["prop"] > 0:
            out["loss_pv_prop"] = self.w["prop"] * bce_dice(self.prop_head(feat), text4, valid4)
        if self.w["skel"] > 0 and skel_heat is not None:
            out["loss_pv_skel"] = self.w["skel"] * bce_dice(self.skel_head(feat), skel_heat.float(), valid4)
        elif self.w["skel"] > 0:
            heats = []
            tnp = text_full.cpu().numpy()
            for m in tnp:
                half = cv2.resize(m.astype(np.uint8), (2 * w, 2 * h), interpolation=cv2.INTER_NEAREST)
                heats.append(skeleton_heatmap(half, out_hw=(h, w))[1] if half.any() else np.zeros((h, w), np.float32))
            heat = torch.from_numpy(np.stack(heats))[:, None].to(feat.device)
            out["loss_pv_skel"] = self.w["skel"] * bce_dice(self.skel_head(feat), heat, valid4)
        if self.w["sdf"] > 0 and word_maps is not None:
            wm = F.interpolate(word_maps[:, None].float(), size=(h, w), mode="nearest")[:, 0].cpu().numpy().astype(np.uint8)
            sdfs, oks = zip(*[word_sdf_from_map(m) for m in wm])
            ok = torch.tensor(oks, device=feat.device)
            pred = self.sdf_head(feat)
            if ok.any():
                tgt = torch.from_numpy(np.stack(sdfs))[:, None].to(feat.device)
                vm = valid4.float() * ok[:, None, None, None].float()
                out["loss_pv_sdf"] = self.w["sdf"] * ((pred.float() - tgt).abs() * vm).sum() / vm.sum().clamp_min(1)
            else:
                out["loss_pv_sdf"] = pred.sum() * 0.0
        return out

    def verify_loss(self, feat, word_maps, word_texts, prompt_words, word_emb, stats=None):
        """feat (B,C,h,w); word_maps (B,H,W) ids (1..n = word_texts[b][i-1]); prompt_words[b] list of K_b
        words with CLIP embeddings word_emb[b] (K_b, text_dim). Candidates = GT words present in the crop."""
        B, C, h, w = feat.shape
        wm = F.interpolate(word_maps[:, None].float(), size=(h, w), mode="nearest")[:, 0].long()
        losses = []
        for b in range(B):
            K = len(prompt_words[b])
            if K == 0:
                continue
            ids = [i for i in wm[b].unique().tolist() if i > 0 and i <= len(word_texts[b])]
            if not ids:
                continue
            onehot = torch.stack([(wm[b] == i).float() for i in ids])                     # (N,h,w)
            pooled = (onehot[:, None] * feat[b][None]).sum((-1, -2)) / onehot.sum((-1, -2))[:, None].clamp_min(1)
            v = F.normalize(self.ver_vis(pooled.float()), dim=-1)
            t = F.normalize(self.ver_txt(word_emb[b].float()), dim=-1)
            logits = self.ver_scale * (v @ t.t()) + self.ver_bias                           # (N,K)
            y = torch.tensor([[float(match_words(word_texts[b][i - 1], pw)) for pw in prompt_words[b]] for i in ids],
                             device=feat.device)
            pos = y.sum()
            if pos == 0:
                continue
            neg = y.numel() - pos
            wgt = torch.where(y > 0, neg.clamp_min(1) / pos, torch.ones_like(y))          # balanced
            losses.append((F.binary_cross_entropy_with_logits(logits, y, reduction="none") * wgt).sum() / wgt.sum())
            if stats is not None:
                has = y.sum(1) > 0
                if has.any():
                    stats["ver_n"] = stats.get("ver_n", 0) + int(has.sum())
                    stats["ver_hit"] = stats.get("ver_hit", 0) + int(
                        y[has].gather(1, logits[has].argmax(1, keepdim=True)).sum())
        if not losses:
            return {"loss_pv_verify": feat.sum() * 0.0 + self.ver_scale * 0.0}
        return {"loss_pv_verify": self.w["ver"] * torch.stack(losses).mean()}
