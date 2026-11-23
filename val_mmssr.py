"""MMSSR-Net validation script with slicing inference.

Supports three test modes:
  --mode dual       : Both transmissive + reflective (default)
  --mode trans      : Transmissive-only (reflective channels filled with 114)
  --mode reflect    : Reflective-only (transmissive channels filled with 114)

Uses SAHI slicing: 1024x1024, 300px overlap, ECP+IoF+rotated NMS.
"""

import argparse
import os

import cv2
import numpy as np
from pathlib import Path

from sahi_infer.sahi_infer import (
    CustomUltralyticsModel,
    AdvancedPostprocessPredictions,
    get_sliced_prediction,
    _draw_predictions_on_reflective_image,
    _save_predictions_for_map,
    collect_rgb_images,
)

SLICE_HW      = 1024
OVERLAP       = 300
OVERLAP_RATIO = OVERLAP / SLICE_HW
FILL_VALUE    = 114  # mean-fill placeholder for missing modality


def rgb_to_t_path(rgb_path):
    """Map reflective image path to its transmissive counterpart."""
    dir_path, filename = os.path.split(rgb_path)
    name, ext = os.path.splitext(filename)
    parts = name.split("-")
    if parts[-1] != "1":
        return None
    parts[-1] = "2"
    return os.path.join(dir_path, "-".join(parts) + ext)


def build_6ch_input(img_r, img_t, mode="dual"):
    """Build 6-channel input tensor according to the test mode.

    Channel convention: [0:3]=transmissive, [3:6]=reflective
    """
    if mode == "dual":
        return np.concatenate((img_t, img_r), axis=-1)

    if mode == "trans":
        # Transmissive only: fill reflective channels
        filled = np.full_like(img_r, FILL_VALUE, dtype=np.uint8)
        return np.concatenate((img_t, filled), axis=-1)

    if mode == "reflect":
        # Reflective only: fill transmissive channels
        filled = np.full_like(img_t, FILL_VALUE, dtype=np.uint8)
        return np.concatenate((filled, img_r), axis=-1)

    raise ValueError(f"Unknown mode: {mode}. Use 'dual', 'trans', or 'reflect'.")


def main():
    parser = argparse.ArgumentParser(description="MMSSR-Net validation with slicing")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--img_dir", required=True, help="Directory with reflective images")
    parser.add_argument("--output", default="runs/val_mmssr", help="Output directory")
    parser.add_argument("--device", default="0", help="Device: 0, cpu, etc.")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--mode", default="dual", choices=["dual", "trans", "reflect"],
                        help="Test mode: dual (both), trans (transmissive-only), reflect (reflective-only)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Test mode: {args.mode}")

    model = CustomUltralyticsModel(
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=args.device,
    )

    postprocess = AdvancedPostprocessPredictions(
        match_threshold=0.5,
        use_advanced_combiner=True,
        edge_penalty_ratio=0.6,
        edge_margin=32,
        iof_threshold=0.85,
        size_ratio_threshold=0.6,
        verbose=False,
    )

    rgb_paths = collect_rgb_images(args.img_dir)
    print(f"Found {len(rgb_paths)} image pairs.")

    total_boxes = 0

    for idx, rgb_path in enumerate(rgb_paths, 1):
        t_path = rgb_to_t_path(rgb_path)
        if t_path is None or not os.path.exists(t_path):
            print(f"  [{idx}/{len(rgb_paths)}] SKIP: no transmissive image for {rgb_path}")
            continue

        img_r = cv2.imread(rgb_path)
        img_t = cv2.imread(t_path)
        if img_r is None or img_t is None:
            print(f"  [{idx}/{len(rgb_paths)}] SKIP: failed to read images")
            continue

        img_6ch = build_6ch_input(img_r, img_t, mode=args.mode)

        result = get_sliced_prediction(
            img_6ch,
            detection_model=model,
            slice_height=SLICE_HW,
            slice_width=SLICE_HW,
            overlap_height_ratio=OVERLAP_RATIO,
            overlap_width_ratio=OVERLAP_RATIO,
            auto_slice_resolution=False,
            perform_standard_pred=False,
            postprocess_class=postprocess,
            verbose=1,
        )

        result.object_prediction_list = [
            p for p in result.object_prediction_list if p.score.value >= args.conf
        ]

        stem = Path(rgb_path).stem
        vis_path = os.path.join(args.output, f"{stem}.png")
        txt_path = os.path.join(args.output, f"{stem}.txt")

        vis = _draw_predictions_on_reflective_image(img_r, result.object_prediction_list)
        cv2.imwrite(vis_path, vis)
        _save_predictions_for_map(txt_path, result.object_prediction_list, img_r.shape[1], img_r.shape[0])

        n = len(result.object_prediction_list)
        total_boxes += n

        stats = postprocess.get_stats()
        print(f"  [{idx}/{len(rgb_paths)}] {stem}: {n} boxes "
              f"(edge={stats.get('edge_penalized',0)}, iof={stats.get('iof_filtered',0)}, nms={stats.get('nms_removed',0)})")

    print(f"\nValidation complete. Total boxes: {total_boxes}. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
