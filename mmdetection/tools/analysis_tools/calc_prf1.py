#!/usr/bin/env python
"""Calculate Precision, Recall and F1 so that they align with COCO mAP.

This script mirrors the IoU thresholds used for mAP in COCO (0.50:0.05:0.95)
and outputs P, R, F1 per threshold as well as their mean values.  It can also
print per-class numbers.

Example
-------
python tools/analysis/calc_prf1.py \
    --gt /path/to/test.json \
    --dt /path/to/results.bbox.json \
    --per-class
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from typing import List

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def prf1(precision: float | np.ndarray, recall: float | np.ndarray, eps: float = 1e-12):
    """Compute F1 from precision & recall with numerical stability."""
    return 2 * precision * recall / (precision + recall + eps)


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute P/R/F1 at COCO IoUs.")
    parser.add_argument("--gt", required=True, type=Path, help="Ground-truth COCO JSON")
    parser.add_argument("--dt", required=True, type=Path, help="Detection results JSON (from MMDet --outfile-prefix)")
    parser.add_argument("--per-class", action="store_true", help="Print per-class metrics as well")
    parser.add_argument("--out", type=Path, help="Optional path to save the metrics text output")
    return parser.parse_args()


def main():
    args = parse_args()

    # Evaluate IoU 0.30 separately (requested), then the standard mAP set
    # of IoU thresholds: 0.50, 0.55, …, 0.95. The 0.30 result will **not**
    # participate in the mean calculation that follows.
    iou_thrs: List[float] = [0.30] + [x / 100 for x in range(50, 100, 5)]

    coco_gt = COCO(str(args.gt))
    coco_dt = coco_gt.loadRes(str(args.dt))

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.iouThrs = np.array(iou_thrs)
    coco_eval.evaluate()
    coco_eval.accumulate()

    precisions = coco_eval.eval["precision"]  # shape: T x R x K x A x M
    recalls = coco_eval.eval["recall"]        # shape: T x K x A x M

    cat_ids = coco_eval.params.catIds
    cat_names = [coco_gt.cats[cid]["name"] for cid in cat_ids]

    # Use area index 0 (all areas) and maxDets index -1 (default 100)
    area_idx = 0
    max_det_idx = -1

    global_p: List[float] = []
    global_r: List[float] = []
    global_f1: List[float] = []

    output_lines: List[str] = []

    title_line = "\n=== Global metrics (aligned with mAP IoUs) ==="
    print(title_line)
    output_lines.append(title_line)
    for t, thr in enumerate(iou_thrs):
        p_t = precisions[t, :, :, area_idx, max_det_idx]
        r_t = recalls[t, :, area_idx, max_det_idx]

        # Filter out invalid entries (marked -1)
        valid_p = p_t[p_t > -1]
        valid_r = r_t[r_t > -1]

        mean_p = float(valid_p.mean()) if valid_p.size else 0.0
        mean_r = float(valid_r.mean()) if valid_r.size else 0.0
        f1 = float(prf1(mean_p, mean_r))

        global_p.append(mean_p)
        global_r.append(mean_r)
        global_f1.append(f1)

        line = f"IoU={thr:.2f}  Precision={mean_p:.4f}  Recall={mean_r:.4f}  F1={f1:.4f}"
        print(line)
        output_lines.append(line)

    # Mean over IoUs 0.50–0.95 (skip the first element, which is 0.30)
    mean_p = np.mean(global_p[1:])
    mean_r = np.mean(global_r[1:])
    mean_f1 = np.mean(global_f1[1:])

    title_mean = "\nMean over IoUs 0.50–0.95 (same idea as mAP):"
    print(title_mean)
    output_lines.append(title_mean)

    mean_line = f"Precision={mean_p:.4f}  Recall={mean_r:.4f}  F1={mean_f1:.4f}"
    print(mean_line)
    output_lines.append(mean_line)

    if args.per_class:
        print("\nPer-class metrics (mean over recall axis):")
        for idx, name in enumerate(cat_names):
            p_cls = []
            r_cls = []
            f1_cls = []
            # Skip index 0 (IoU 0.30) when computing the per-class mean
            for t in range(1, len(iou_thrs)):
                prs = precisions[t, :, idx, area_idx, max_det_idx]
                rec = recalls[t, idx, area_idx, max_det_idx]
                valid_prs = prs[prs > -1]
                mean_p_cls = float(valid_prs.mean()) if valid_prs.size else 0.0
                p_cls.append(mean_p_cls)
                r_cls.append(float(rec))
                f1_cls.append(float(prf1(mean_p_cls, rec)))
            # Average over IoUs to align with mAP
            cls_line = f"{name:<20}  P={np.mean(p_cls):.4f}  R={np.mean(r_cls):.4f}  F1={np.mean(f1_cls):.4f}"
            print(cls_line)
            output_lines.append(cls_line)

    done_line = "Done."
    print(done_line)
    output_lines.append(done_line)

    # ------------------------------------------------------------------
    # Optional file output
    # ------------------------------------------------------------------
    if args.out is not None:
        # Resolve output file path: if user passed a directory (or a path
        # without a suffix), write <dir>/prf1.txt. This prevents
        # IsADirectoryError when a directory path is supplied.
        out_path = args.out
        if out_path.is_dir() or out_path.suffix == "":
            out_path = out_path / "prf1.txt"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(output_lines))
        print(f"\nMetrics written to {out_path}")


if __name__ == "__main__":
    main() 