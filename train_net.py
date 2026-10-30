#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
"""
LaTeX-Seg training script.

Datasets are read from $DETECTRON2_DATASETS (default: ./datasets) with this layout:

    datasets/
      TextSeg/
        train/  test/                 images
        seg_train/  seg_test/         masks (text=255, background=0)
        train_gpt/  test_gpt/         LLM prompts, one <image-stem>.txt per image
      HierText/
        train/  test/                 images
        train_gt/  test_gt/           masks
        train_gemini/  test_gemini/   LLM prompts

Each prompt file has 3 lines: location / style / recognition.
"""
import os
import random
import re
import sys
import time

import cv2
import natsort
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this folder's latex_seg

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.utils.events import get_event_storage
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import SemSegEvaluator, verify_results
from detectron2.evaluation.sem_seg_evaluation import load_image_into_numpy_array
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping

from latex_seg import ColorAugSSDTransform, add_latexseg_config


# =====================================================================
# Datasets
# =====================================================================
def get_textseg_sem(img_dir, mask_dir):
    # image 파일만 (train 폴더에 섞인 a00001_output 같은 디렉터리 제외)
    img_list = natsort.natsorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])
    sem_seg_file_list = natsort.natsorted(os.listdir(mask_dir))
    dataset_dicts = []
    for idx, v in enumerate(img_list):
        filename = os.path.join(img_dir, v)
        height, width = cv2.imread(filename).shape[:2]
        dataset_dicts.append({
            "file_name": filename,
            "image_id": idx,
            "height": height,
            "width": width,
            "sem_seg_file_name": os.path.join(mask_dir, sem_seg_file_list[idx]),
        })
    return dataset_dicts


_DATASET_ROOT = os.getenv("DETECTRON2_DATASETS", "datasets")

# name -> (image dir, mask dir, prompt dir), relative to _DATASET_ROOT
_DATASETS = {
    "Textseg_sem_train":  ("TextSeg/train", "TextSeg/seg_train", "TextSeg/train_gpt"),
    "Textseg_sem_test":   ("TextSeg/test", "TextSeg/seg_test", "TextSeg/test_gpt"),
    "Textseg_sem_valid":  ("TextSeg/valid", "TextSeg/seg_valid", "TextSeg/valid_gpt"),
    "HierText_sem_train": ("HierText/train", "HierText/train_gt", "HierText/train_gemini"),
    "HierText_sem_test":  ("HierText/test", "HierText/test_gt", "HierText/test_gemini"),
}


def register_text_datasets(root):
    for name, (img, seg, prompt) in _DATASETS.items():
        img_dir, seg_dir = os.path.join(root, img), os.path.join(root, seg)
        DatasetCatalog.register(name, lambda i=img_dir, s=seg_dir: get_textseg_sem(i, s))
        MetadataCatalog.get(name).set(
            stuff_classes=["Background", "Text"],
            evaluator_type="sem_seg",
            ignore_label=255,
            prompt_dir=os.path.join(root, prompt),
        )


register_text_datasets(_DATASET_ROOT)


def _strip_prefix(line):
    line = line.strip()
    if len(line) >= 2 and (line[1] == "." or line[1] == ")") and line[0].isdigit():
        return line[2:].strip()
    return line


def load_binary_text_mask(filename, dtype=None):
    # GT masks are 0/255 and 255 is also the ignore label: binarize (text=1) as the mapper does
    return (load_image_into_numpy_array(filename) > 127).astype(dtype or int)


def read_prompt_triplet(txt_path):
    """Mirror Hi-SAM's _read_text_triplet: 3 non-empty lines ->
    (location, style, semantic); '1./2./3.' numbering stripped; any extra
    lines are appended to the semantic string."""
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"Text prompt file not found: {txt_path}")
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    loc = _strip_prefix(lines[0]) if len(lines) > 0 else ""
    sty = _strip_prefix(lines[1]) if len(lines) > 1 else ""
    sem = _strip_prefix(lines[2]) if len(lines) > 2 else ""
    if len(lines) > 3:
        sem = " ".join([sem] + [x.strip() for x in lines[3:]])
    return loc, sty, sem


_NUMBERED = re.compile(r"(?:^|\n)\s*([123])[.)]\s*")


def parse_prompt_lines(txt_path):
    """Strict parse for prompt mixing: (location, style, recognition) or None per line when the line is missing or
    degenerate (VLM failure modes seen in train_qwen3vl / train_deepseek: coordinate dumps, number loops,
    two items on one line)."""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return (None, None, None)
    parts = _NUMBERED.split(text)  # ["", "1", body, "2", body, ...]
    found = {}
    for k in range(1, len(parts) - 1, 2):
        found.setdefault(int(parts[k]), " ".join(parts[k + 1].split()))
    out = []
    for i in (1, 2, 3):
        t = found.get(i)
        toks = t.split() if t else []
        bad = (not t or len(t) > 4000 or re.search(r"\b[123][.)] The text", t)
               or (i == 1 and sum(c.isdigit() for c in t) > 20)
               or (len(toks) > 10 and sum(w.strip(",.;").isdigit() for w in toks) > 0.5 * len(toks)))
        out.append(None if bad else t)
    return tuple(out)


class PromptMixer:
    """Draws the training prompt of an image from several LLM prompt dirs (cfg.INPUT.PROMPT_MIX).
    Malformed lines are skipped; if no source has a valid line, the dataset's own prompt file is used as is."""

    def __init__(self, own_dir, sources, weights=(), per_line=False):
        root = os.path.dirname(own_dir.rstrip("/"))
        self.own_dir = own_dir
        self.dirs = [os.path.join(root, s) for s in sources]
        for d in self.dirs:
            assert os.path.isdir(d), f"PROMPT_MIX source not found: {d}"
        self.weights = list(weights) if weights else [1.0] * len(self.dirs)
        assert len(self.weights) == len(self.dirs), (weights, sources)
        self.per_line = per_line
        self._cache = {}

    def _lines(self, stem):
        if stem not in self._cache:
            self._cache[stem] = [parse_prompt_lines(os.path.join(d, stem + ".txt")) for d in self.dirs]
        return self._cache[stem]

    def _draw(self, cands):
        return random.choices([c for c, _ in cands], weights=[w for _, w in cands])[0]

    def __call__(self, stem):
        lines = self._lines(stem)
        own = None
        if self.per_line:
            out = []
            for i in range(3):
                cands = [(ls[i], w) for ls, w in zip(lines, self.weights) if ls[i] is not None]
                if cands:
                    out.append(self._draw(cands))
                else:
                    own = own or read_prompt_triplet(os.path.join(self.own_dir, stem + ".txt"))
                    out.append(own[i])
            return tuple(out)
        cands = [(ls, w) for ls, w in zip(lines, self.weights) if None not in ls]
        if cands:
            return tuple(self._draw(cands))
        return read_prompt_triplet(os.path.join(self.own_dir, stem + ".txt"))


class TextPromptDatasetMapper(DatasetMapper):
    """DatasetMapper that also attaches location/style/recognition text
    (read from `<prompt_dir>/<image-stem>.txt`, or drawn by a PromptMixer) to each sample."""

    prompt_mix = None

    def set_prompt_dir(self, prompt_dir):
        self.prompt_dir = prompt_dir
        return self

    def set_prompt_mix(self, mix_cfg):
        if mix_cfg.SOURCES:
            self.prompt_mix = PromptMixer(self.prompt_dir, mix_cfg.SOURCES, mix_cfg.WEIGHTS, mix_cfg.PER_LINE)
        return self

    def read_prompt(self, stem):
        if self.prompt_mix is not None:
            return self.prompt_mix(stem)
        return read_prompt_triplet(os.path.join(self.prompt_dir, stem + ".txt"))

    def __call__(self, dataset_dict):
        stem = os.path.splitext(os.path.basename(dataset_dict["file_name"]))[0]
        out = super().__call__(dataset_dict)
        # GT 마스크가 0/255인데 ignore_label도 255 -> 텍스트=1, 배경=0으로 이진화
        if "sem_seg" in out:
            out["sem_seg"] = (out["sem_seg"] > 127).long()
        loc, sty, rec = self.read_prompt(stem)
        out["location_text"] = loc
        out["style_text"] = sty
        out["recognition_text"] = rec
        return out


class PVTextPromptDatasetMapper(TextPromptDatasetMapper):
    """Training mapper for MODEL.PV: also builds the TextSeg word instance map from <ANNO_DIR>/<stem>_anno.json,
    transforms it with the same augmentations as the mask (nearest), and the skeleton heat target at stride 4."""

    def set_anno_dir(self, anno_dir):
        self.anno_dir = anno_dir
        return self

    def __call__(self, dataset_dict):
        import copy
        import numpy as np
        from detectron2.data import detection_utils as utils
        from latex_seg.pv_heads import load_word_quads, word_map_from_quads
        from latex_seg.skeleton_target import skeleton_heatmap

        dataset_dict = copy.deepcopy(dataset_dict)
        stem = os.path.splitext(os.path.basename(dataset_dict["file_name"]))[0]
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        sem = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        ap = os.path.join(self.anno_dir, stem + "_anno.json")
        quads = load_word_quads(ap) if os.path.exists(ap) else []
        wm = word_map_from_quads(quads, *image.shape[:2])
        aug_input = T.AugInput(image, sem_seg=sem)
        transforms = self.augmentations(aug_input)
        image, sem = aug_input.image, aug_input.sem_seg
        wm = transforms.apply_segmentation(wm)
        out = dataset_dict
        out["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        out["sem_seg"] = (torch.as_tensor(sem.astype("long")) > 127).long()
        out["word_map"] = torch.as_tensor(wm.astype("long"))
        out["word_texts"] = [t for t, _ in quads]
        H, W = sem.shape
        h, w = (H + 3) // 4, (W + 3) // 4
        txt = (sem > 127).astype(np.uint8)
        half = cv2.resize(txt, (2 * w, 2 * h), interpolation=cv2.INTER_NEAREST)
        heat = skeleton_heatmap(half, out_hw=(h, w))[1] if half.any() else np.zeros((h, w), np.float32)
        out["skel_heat"] = torch.from_numpy(heat)
        loc, sty, rec = self.read_prompt(stem)
        out["location_text"], out["style_text"], out["recognition_text"] = loc, sty, rec
        return out


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    if cfg.INPUT.COLOR_AUG_SSD:
        augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
    augs.append(T.RandomFlip())
    return augs


# =====================================================================
# Text-consistency loss (Hi-SAM --cons_mode safe --cons_help)
# =====================================================================
_TEXT_KEYS = ("location_text", "style_text", "recognition_text",
              "location_prompt_tokens", "style_prompt_tokens", "recognition_prompt_tokens")

# Text-path parameters: the only ones the help term updates.
_TEXT_PATH_PREFIXES = ("text_model.", "text_adapters.", "text_fusion.text_adapters.")


class TextConsistencyMixin:
    """Hi-SAM의 text-consistency loss (--cons_mode safe --cons_help)를 detectron2 학습 루프로 이식.

    매 step 배치 일부(NEG_BATCH)에 대해 3가지 텍스트 조건으로 forward 한다.
      correct : 정답 텍스트
      no-text : 텍스트 없음 (모델에서 0 토큰으로 처리)
      wrong   : FIFO 버퍼의 과거 텍스트
    Q = coarse 분할 손실 (Hi-SAM consistency_on='lr' 에 대응), sg = stop-gradient.
      no-text seg : NOTEXT_SEG_WEIGHT * Seg(no-text)                        (전체 파라미터)
      safe        : SAFE_WEIGHT * relu(Q(wrong) - sg[Q(no-text)])            (전체 파라미터)
      help        : HELP_WEIGHT * relu(Q(correct) - sg[Q(no-text)] + HELP_MARGIN)
                    or, with HELP_REL_MARGIN > 0, HELP_WEIGHT * relu(Q(correct) - (1 - HELP_REL_MARGIN) * sg[Q(no-text)])
                    (텍스트 경로 파라미터에만 gradient; 별도 backward)
      content     : CONTENT_WEIGHT * mean_i relu(Q_i(correct) - (1 - RHO) * sg[Q_i(wrong)])
                    (이미지별 hinge 후 평균; help 와 같은 텍스트 경로 파라미터에만 gradient)
    help / safe 는 배치(NEG_BATCH) 평균 Q 에 한 번 hinge, content 는 이미지별 Q.
    """
    TC_ENABLED = False
    TC_NOTEXT_SEG_WEIGHT = 1.0
    TC_SAFE_WEIGHT = 0.1
    TC_HELP_WEIGHT = 0.5
    TC_HELP_MARGIN = 0.2
    TC_BUFFER = 64
    TC_NEG_BATCH = 2
    TC_HELP_REL_MARGIN = 0.0
    TC_CONTENT_WEIGHT = 0.0
    TC_CONTENT_RHO = 0.1
    _tc_buf = None
    _tc_stats_file = None

    def _tc_backward(self, data):
        """3가지 조건 forward 를 하고 gradient 를 .grad 에 누적한다. 메인 손실 backward 뒤에 호출.
        메모리를 아끼려고 (no-text + wrong) 과 help 를 차례로 계산하고 그래프를 바로 해제한다.
        gradient 합은 모든 손실을 더해 한 번에 backward 한 것과 같다. Returns dict for logging."""
        if not self.TC_ENABLED or "location_text" not in data[0]:
            return {}
        if self._tc_buf is None:
            self._tc_buf = []
        for x in data:
            self._tc_buf.append((x["file_name"], (x["location_text"], x["style_text"], x["recognition_text"])))
        while len(self._tc_buf) > self.TC_BUFFER:
            self._tc_buf.pop(0)
        if len(self._tc_buf) < 8:
            return {}

        # 메모리 제약: 3가지 조건 forward 는 배치 일부만 사용. 정방향 학습은 전체 배치.
        k = min(self.TC_NEG_BATCH, len(data))
        sub = [data[i] for i in random.sample(range(len(data)), k)]
        notext = [{key: v for key, v in x.items() if key not in _TEXT_KEYS} for x in sub]
        wrong = []
        for x in sub:
            d = dict(x)
            # never the image's own prompt (from any LLM source)
            others = [t for f, t in self._tc_buf if f != x["file_name"]]
            d["location_text"], d["style_text"], d["recognition_text"] = random.choice(others)
            wrong.append(d)

        model = self._unwrapped_model()
        coarse = model.sem_seg_head.coarse_sem_seg_head
        model.aux_losses_off = True   # PV / role-head losses only on the main forward
        coarse.per_image_q = True     # also return the per-image Q ("q_img")
        try:
            return self._tc_forwards(model, sub, notext, wrong)
        finally:
            model.aux_losses_off = False
            coarse.per_image_q = False

    def _tc_forwards(self, model, sub, notext, wrong):
        # (1) no-text seg + safe: 전체 파라미터
        nt_losses = model(notext)
        nt_q = nt_losses["loss_sem_seg"].detach()
        nt_q_img = nt_losses["q_img"].detach()
        nt_seg = self.TC_NOTEXT_SEG_WEIGHT * (nt_losses["loss_sem_seg"] + nt_losses.get("loss_igr", 0.0))
        wr_losses = model(wrong)
        wr_q, wr_q_img = wr_losses["loss_sem_seg"], wr_losses["q_img"].detach()
        safe = self.TC_SAFE_WEIGHT * torch.relu(wr_q - nt_q)
        self._add_grads(nt_seg + safe, [p for p in model.parameters() if p.requires_grad])
        # (2) help + content: 텍스트 경로 파라미터만
        co_losses = model(sub)
        co_q, co_q_img = co_losses["loss_sem_seg"], co_losses["q_img"]
        help_term = self.TC_HELP_WEIGHT * torch.relu(co_q - self._help_target(nt_q))
        content_img = torch.relu(co_q_img - (1.0 - self.TC_CONTENT_RHO) * wr_q_img)
        content = self.TC_CONTENT_WEIGHT * content_img.mean()
        text_term = help_term + content
        if text_term.item() > 0.0:
            self._add_grads(text_term, self._text_path_params())
        co_q_img = co_q_img.detach()
        stats = {
            "tc_q_correct": co_q.item(), "tc_q_notext": nt_q.item(), "tc_q_wrong": wr_q.item(),
            "tc_q_correct_img": co_q_img.mean().item(), "tc_q_notext_img": nt_q_img.mean().item(),
            "tc_q_wrong_img": wr_q_img.mean().item(),
            # help / safe hinge on the batch Q (0 or 1) and, for reference, per image
            "tc_on_help": float(help_term.item() > 0), "tc_on_safe": float(safe.item() > 0),
            "tc_on_help_img": (co_q_img - self._help_target(nt_q_img) > 0).float().mean().item(),
            "tc_on_safe_img": (wr_q_img - nt_q_img > 0).float().mean().item(),
            "tc_on_content_img": (content_img > 0).float().mean().item(),
        }
        self._log_tc_stats(stats, help_term.item(), content.item(), safe.item())
        return {
            "loss_notext_seg": nt_seg.detach(),
            "loss_cons_safe": safe.detach(),
            "loss_cons_help": help_term.detach(),
            "loss_cons_content": content.detach(),
        }

    def _help_target(self, nt_q):
        """help hinge threshold: absolute (Q(no-text) - margin) or relative ((1 - rel) * Q(no-text))."""
        if self.TC_HELP_REL_MARGIN > 0:
            return (1.0 - self.TC_HELP_REL_MARGIN) * nt_q
        return nt_q - self.TC_HELP_MARGIN

    def _log_tc_stats(self, stats, help_v, content_v, safe_v):
        """metrics.json (via the event storage) + every iteration to <OUTPUT_DIR>/tc_stats.tsv."""
        get_event_storage().put_scalars(smoothing_hint=False, **stats)
        if not comm.is_main_process():
            return
        row = dict(iter=self._trainer.iter, help=help_v, content=content_v, safe=safe_v, **stats)
        if self._tc_stats_file is None:
            path = os.path.join(self.cfg.OUTPUT_DIR, "tc_stats.tsv")
            new = not os.path.exists(path)
            self._tc_stats_file = open(path, "a", buffering=1)
            if new:
                self._tc_stats_file.write("\t".join(row) + "\n")
        self._tc_stats_file.write("\t".join(f"{v:.5g}" if isinstance(v, float) else str(v)
                                             for v in row.values()) + "\n")

    def _unwrapped_model(self):
        m = self._trainer.model
        return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m

    def _text_path_params(self):
        return [p for n, p in self._unwrapped_model().named_parameters()
                if p.requires_grad and n.startswith(_TEXT_PATH_PREFIXES)]

    @staticmethod
    def _add_grads(loss, params):
        """loss 의 gradient 를 params 의 .grad 에만 더한다 (멀티 GPU면 평균)."""
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        world = comm.get_world_size()
        for p, g in zip(params, grads):
            if g is None:
                continue
            if world > 1:
                torch.distributed.all_reduce(g)
                g = g / world
            p.grad = g if p.grad is None else p.grad + g


def _patch_run_step(trainer):
    """SimpleTrainer.run_step 을 감싸 consistency loss 를 더한다."""
    inner = trainer._trainer  # SimpleTrainer or AMPTrainer

    def run_step():
        assert inner.model.training
        start = time.perf_counter()
        data = next(inner._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = inner.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        inner.optimizer.zero_grad()
        losses.backward()
        loss_dict = dict(loss_dict, **trainer._tc_backward(data))
        inner._write_metrics(loss_dict, data_time)
        inner.optimizer.step()

    inner.run_step = run_step


# =====================================================================
# Trainer
# =====================================================================
class Trainer(TextConsistencyMixin, DefaultTrainer):
    @classmethod
    def build_optimizer(cls, cfg, model):
        if cfg.SOLVER.OPTIMIZER == "SGD":
            return super().build_optimizer(cfg, model)
        assert cfg.SOLVER.OPTIMIZER == "ADAMW", cfg.SOLVER.OPTIMIZER
        # no weight decay on norm layers and biases
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=0.0,
            weight_decay_bias=0.0,
        )
        clip = cfg.SOLVER.CLIP_GRADIENTS
        if clip.ENABLED and clip.CLIP_TYPE == "full_model":
            # clip the global grad norm over all parameters (as in Mask2Former), not per-parameter
            class FullModelClippingAdamW(torch.optim.AdamW):
                def step(self, closure=None):
                    all_params = [p for g in self.param_groups for p in g["params"]]
                    torch.nn.utils.clip_grad_norm_(all_params, clip.CLIP_VALUE, clip.NORM_TYPE)
                    return super().step(closure=closure)

            return FullModelClippingAdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        return maybe_add_gradient_clipping(
            cfg, torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return SemSegEvaluator(
            dataset_name, distributed=True, output_dir=output_folder,
            sem_seg_loading_fn=load_binary_text_mask,
        )

    @classmethod
    def build_train_loader(cls, cfg):
        prompt_dir = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).prompt_dir
        if cfg.MODEL.PV.ENABLED or cfg.MODEL.ROLE_HEADS.ENABLED:
            mapper = PVTextPromptDatasetMapper(
                cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg)
            ).set_prompt_dir(prompt_dir).set_prompt_mix(cfg.INPUT.PROMPT_MIX).set_anno_dir(cfg.MODEL.PV.ANNO_DIR)
        else:
            mapper = TextPromptDatasetMapper(
                cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg)
            ).set_prompt_dir(prompt_dir).set_prompt_mix(cfg.INPUT.PROMPT_MIX)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # Attach the same text prompts at eval so the trained text branches are used.
        prompt_dir = MetadataCatalog.get(dataset_name).prompt_dir
        mapper = TextPromptDatasetMapper(cfg, is_train=False).set_prompt_dir(prompt_dir)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


def setup(args):
    cfg = get_cfg()
    add_latexseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    tc = cfg.TEXT_CONSISTENCY
    Trainer.TC_ENABLED = tc.ENABLED
    Trainer.TC_NOTEXT_SEG_WEIGHT = tc.NOTEXT_SEG_WEIGHT
    Trainer.TC_SAFE_WEIGHT = tc.SAFE_WEIGHT
    Trainer.TC_HELP_WEIGHT = tc.HELP_WEIGHT
    Trainer.TC_HELP_MARGIN = tc.HELP_MARGIN
    Trainer.TC_BUFFER = tc.BUFFER
    Trainer.TC_NEG_BATCH = tc.NEG_BATCH
    Trainer.TC_HELP_REL_MARGIN = tc.HELP_REL_MARGIN
    Trainer.TC_CONTENT_WEIGHT = tc.CONTENT_WEIGHT
    Trainer.TC_CONTENT_RHO = tc.CONTENT_RHO
    if tc.ENABLED:
        _patch_run_step(trainer)
        print(f"[text-consistency] notext_seg={tc.NOTEXT_SEG_WEIGHT} safe={tc.SAFE_WEIGHT} "
              f"help={tc.HELP_WEIGHT} (margin {tc.HELP_MARGIN}, rel {tc.HELP_REL_MARGIN}) content={tc.CONTENT_WEIGHT} "
              f"(rho {tc.CONTENT_RHO}) buffer={tc.BUFFER}")
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
