"""BiSSR-Net validation script with synchronized bimodal slicing inference.

Uses SAHI-based slicing: 1024x1024 patches, 300px overlap, ECP edge penalty,
IoF containment filtering, and rotated NMS for final de-duplication.
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

SLICE_HW  = 1024
OVERLAP   = 300
OVERLAP_RATIO = OVERLAP / SLICE_HW

def rgb_to_t_path(rgb_path):
    """Map reflective image path to its transmissive counterpart."""
    dir_path, filename = os.path.split(rgb_path)
    name, ext = os.path.splitext(filename)
    parts = name.split("-")
    if parts[-1] != "1":
        return None
    parts[-1] = "2"
    return os.path.join(dir_path, "-".join(parts) + ext)


def main():
    parser = argparse.ArgumentParser(description="BiSSR-Net validation with slicing")
    parser.add_argument("--weights", required=True, help="Path to best.pt")
    parser.add_argument("--img_dir", required=True, help="Directory with reflective images")
    parser.add_argument("--output", default="runs/val_bissr", help="Output directory")
    parser.add_argument("--device", default="0", help="Device: 0, cpu, etc.")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

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
    print(f"Found {len(rgb_paths)} reflective/transmissive image pairs.")

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

        img_6ch = np.concatenate((img_t, img_r), axis=-1)  # [0:3]=transmissive, [3:6]=reflective

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

        stats = postprocess.get_stats()
        print(f"  [{idx}/{len(rgb_paths)}] {stem}: {len(result.object_prediction_list)} boxes "
              f"(edge={stats.get('edge_penalized',0)}, iof={stats.get('iof_filtered',0)}, nms={stats.get('nms_removed',0)})")

    print(f"\nValidation complete. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
