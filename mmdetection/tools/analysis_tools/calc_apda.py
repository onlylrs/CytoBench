#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast segmentation metric evaluator (AP50 / mAP / Dice / AJI + bootstrap CI).

使用示例
--------
python calc_apda_fast.py \
    --gt /path/to/test.json \
    --dt /path/to/results.segm.json \
    --bootstrap 1000 \
    --n-jobs 8
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


# --------------------------------------------------------------------- #
# ---------------------------  utils  --------------------------------- #
# --------------------------------------------------------------------- #
def decode(seg, h: int, w: int) -> np.ndarray:
    """COCO polygon / RLE → bool mask (H, W)."""
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, h, w)
        seg = mask_utils.merge(rles)
    return mask_utils.decode(seg).astype(bool)


def compute_aji_single(
    gt_masks: List[np.ndarray], pr_masks: List[np.ndarray], eps: float = 1e-12
) -> float:
    """Aggregated Jaccard Index for one image (GT-driven greedy map)."""
    if not gt_masks and not pr_masks:
        return 1.0
    if not gt_masks or not pr_masks:
        return 0.0

    g_n, p_n = len(gt_masks), len(pr_masks)
    ious = np.zeros((g_n, p_n), dtype=np.float32)
    for i, g in enumerate(gt_masks):
        for j, p in enumerate(pr_masks):
            inter = (g & p).sum()
            uni = (g | p).sum() + eps
            ious[i, j] = inter / uni

    matched_p = np.full(p_n, False)
    inter_tot, union_tot = 0.0, 0.0

    for gi, g in enumerate(gt_masks):
        pj = ious[gi].argmax()
        if ious[gi, pj] > 0:
            matched_p[pj] = True
            p = pr_masks[pj]
            inter_tot += (g & p).sum()
            union_tot += (g | p).sum()
        else:
            union_tot += g.sum()

    # unmatched pred masks
    for pj, p in enumerate(pr_masks):
        if not matched_p[pj]:
            union_tot += p.sum()

    return inter_tot / (union_tot + eps)


def img_stats_worker(
    img_id: int, coco_gt: COCO, coco_dt: COCO
) -> Dict[str, float]:
    img_info = coco_gt.imgs[img_id]
    h, w = img_info["height"], img_info["width"]

    g_mask = np.zeros((h, w), dtype=bool)
    gt_inst = []
    for ann in coco_gt.imgToAnns[img_id]:
        if ann.get("iscrowd", 0):
            continue
        m = decode(ann["segmentation"], h, w)
        g_mask |= m
        gt_inst.append(m)

    p_mask = np.zeros((h, w), dtype=bool)
    pr_inst = []
    for ann in coco_dt.imgToAnns.get(img_id, []):
        m = decode(ann["segmentation"], h, w)
        p_mask |= m
        pr_inst.append(m)

    inter = (g_mask & p_mask).sum()
    return dict(
        id=img_id,
        gt_area=g_mask.sum(),
        pr_area=p_mask.sum(),
        inter=inter,
        aji=compute_aji_single(gt_inst, pr_inst),
    )


# --------------------------------------------------------------------- #
# ---------------------------  main  ---------------------------------- #
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Fast segmentation metric evaluator")
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--dt", required=True, type=Path)
    p.add_argument("--bootstrap", type=int, default=0, metavar="N")
    p.add_argument("--n-jobs", type=int, default=os.cpu_count())
    p.add_argument("--out", type=Path)
    p.add_argument("--per-class", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    coco_gt = COCO(str(args.gt))
    coco_dt = coco_gt.loadRes(str(args.dt))

    iou_thrs = np.arange(0.5, 0.96, 0.05)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.params.iouThrs = iou_thrs
    coco_eval.evaluate()
    coco_eval.accumulate()

    # ---------- 一次性预解码，缓存 per-image stats ----------
    img_ids = coco_gt.getImgIds()
    print(f"Precomputing per-image masks on {args.n_jobs} CPU cores …")
    stats: List[Dict[str, float]] = Parallel(args.n_jobs)(
        delayed(img_stats_worker)(img_id, coco_gt, coco_dt) for img_id in tqdm(img_ids)
    )
    # list -> numpy for vectorised bootstrap
    gt_area = np.array([s["gt_area"] for s in stats])
    pr_area = np.array([s["pr_area"] for s in stats])
    inter_area = np.array([s["inter"] for s in stats])
    aji_arr = np.array([s["aji"] for s in stats])

    # ----------------- baseline metrics -------------------
    precision = coco_eval.eval["precision"]  # T×R×K×A×M
    area_idx, max_det_idx = 0, -1
    ap50 = precision[0, :, :, area_idx, max_det_idx]
    ap50 = ap50[ap50 > -1].mean()
    ap_all = []
    for t in range(len(iou_thrs)):
        p_t = precision[t, :, :, area_idx, max_det_idx]
        p_t = p_t[p_t > -1]
        ap_all.append(p_t.mean() if p_t.size else 0)
    mAP = float(np.mean(ap_all))

    dice = 2 * inter_area.sum() / (gt_area.sum() + pr_area.sum() + 1e-12)
    AJI = float(aji_arr.mean())

    print("\n=== Global metrics ===")
    print(f"AP50 : {ap50:.6f}")
    print(f"mAP  : {mAP:.6f}")
    print(f"Dice : {dice:.6f}")
    print(f"AJI  : {AJI:.6f}")

    out_lines = [
        "=== Global metrics ===",
        f"AP50 : {ap50:.6f}",
        f"mAP  : {mAP:.6f}",
        f"Dice : {dice:.6f}",
        f"AJI  : {AJI:.6f}",
    ]

    # ------------------ bootstrap CI ----------------------
    if args.bootstrap > 0:
        print(f"\nBootstrapping {args.bootstrap}× …")
        rng = np.random.default_rng()
        n_img = len(img_ids)

        # 缓存 evalImgs / index map
        orig_evalImgs = coco_eval.evalImgs
        n_cats = len(coco_eval.params.catIds)
        n_area = len(coco_eval.params.areaRng)
        id_to_idx = {img_id: i for i, img_id in enumerate(img_ids)}

        def make_evalImgs(sample_ids: List[int]) -> list:
            """按新顺序重排 evalImgs (cat, area, img) flattened list."""
            new_eval = []
            for c in range(n_cats):
                for a in range(n_area):
                    for img_id in sample_ids:
                        new_eval.append(
                            orig_evalImgs[(c * n_area + a) * n_img + id_to_idx[img_id]]
                        )
            return new_eval

        ap50_s, map_s, dice_s, aji_s = [], [], [], []

        for _ in tqdm(range(args.bootstrap), unit="resample"):
            sample_ids = rng.choice(img_ids, size=n_img, replace=True).tolist()

            # ---- fast AP (reuse IoU) ----
            coco_eval.params.imgIds = sample_ids
            coco_eval.evalImgs = make_evalImgs(sample_ids)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval.accumulate()

            prec_bs = coco_eval.eval["precision"]
            ap50_bs = prec_bs[0, :, :, area_idx, max_det_idx]
            ap50_bs = ap50_bs[ap50_bs > -1].mean()
            ap_all_bs = []
            for t in range(len(iou_thrs)):
                p_t = prec_bs[t, :, :, area_idx, max_det_idx]
                p_t = p_t[p_t > -1]
                ap_all_bs.append(p_t.mean() if p_t.size else 0)
            map_bs = float(np.mean(ap_all_bs))

            # ---- fast Dice/AJI (vectorised) ----
            mask = np.isin(img_ids, sample_ids)
            dice_bs = 2 * inter_area[mask].sum() / (
                gt_area[mask].sum() + pr_area[mask].sum() + 1e-12
            )
            aji_bs = aji_arr[mask].mean()

            ap50_s.append(ap50_bs)
            map_s.append(map_bs)
            dice_s.append(dice_bs)
            aji_s.append(aji_bs)

        def ci(x):  # 95 %
            return np.percentile(x, [2.5, 97.5])

        print("\n--- 95% CI ---")
        for name, arr in zip(
            ["AP50", "mAP", "Dice", "AJI"], [ap50_s, map_s, dice_s, aji_s]
        ):
            lo, hi = ci(arr)
            print(f"{name:<5}: [{lo:.6f}, {hi:.6f}]")
            out_lines.append(f"{name:<5}: [{lo:.6f}, {hi:.6f}]")
        # ----------------  concise summary line ----------------
    summary = (
        f"\nSUMMARY\n"
        f"mAP : {mAP:.6f} ({ci(map_s)[0]:.6f}, {ci(map_s)[1]:.6f}) \n"
        f"Dice: {dice:.6f} ({ci(dice_s)[0]:.6f}, {ci(dice_s)[1]:.6f}) \n"
        f"AJI : {AJI:.6f} ({ci(aji_s)[0]:.6f}, {ci(aji_s)[1]:.6f}) \n"
        f"AP50: {ap50:.6f} ({ci(ap50_s)[0]:.6f}, {ci(ap50_s)[1]:.6f})"
    )
    print(summary)
    out_lines.append(summary)

    # ------------- per-class AP（可选） ------------------
    if args.per_class:
        print("\nPer-class AP50 / mAP:")
        out_lines.append("\nPer-class AP50 / mAP:")
        cat_ids = coco_gt.getCatIds()
        cat_names = [coco_gt.cats[c]["name"] for c in cat_ids]
        for cid, name in zip(cat_ids, cat_names):
            img_cids = coco_gt.getImgIds(catIds=[cid])
            if not img_cids:
                continue
            coco_eval_c = COCOeval(coco_gt, coco_dt, "segm")
            coco_eval_c.params.imgIds = img_cids
            coco_eval_c.params.catIds = [cid]
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval_c.evaluate()
                coco_eval_c.accumulate()
            p = coco_eval_c.eval["precision"]
            ap50_c = p[0, :, 0, area_idx, max_det_idx]
            ap50_c = ap50_c[ap50_c > -1].mean()
            ap_all_c = [
                p[t, :, 0, area_idx, max_det_idx][
                    p[t, :, 0, area_idx, max_det_idx] > -1
                ].mean()
                for t in range(len(iou_thrs))
            ]
            map_c = float(np.mean(ap_all_c))
            print(f"{name:<20} AP50={ap50_c:.4f}  mAP={map_c:.4f}")
            out_lines.append(f"{name:<20} AP50={ap50_c:.4f}  mAP={map_c:.4f}")

    # ----------------------- save ------------------------
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(out_lines))
        print(f"\nMetrics saved to {args.out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
