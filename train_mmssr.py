"""MMSSR-Net training script.

MMSSR-Net uses Progressive Mixed-Modality Training (RG-PMMT) to learn
robust cross-modal representations that work with dual, transmissive-only,
and reflective-only inputs.

Copy the modified files (tasks.py, base.py, DEA.py) into the ultralytics
package before running, or place them directly in this project's
ultralytics/ directory so they override the originals.
"""

from ultralytics import YOLO

# ─── User config ───────────────────────────────────────────────
MODEL_CFG = "ultralytics/cfg/models/fuse/dea-seed-fuse.yaml"    # MMSSR-Net model YAML
DATA_YAML = "ultralytics/cfg/datasets/obb-seed.yaml"            # Dataset YAML
SCHEDULE  = "conservative"   # "conservative" | "full_warmup_120" | "legacy"
IMG_SZ    = 640
BATCH     = 16
EPOCHS    = 300
DEVICE    = "0"             # "0" | "0,1,2,3" | "cpu"
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = YOLO(MODEL_CFG)
    model.train(
        data=DATA_YAML,
        ch=6,
        imgsz=IMG_SZ,
        epochs=EPOCHS,
        batch=BATCH,
        workers=8,
        device=DEVICE,
        optimizer="SGD",
        patience=40,
        amp=False,
        cache="disk",
        project="runs/train_mmssr",
        name="mmssr_net",
        resume=False,
        fraction=1.0,
        # ─── RG-PMMT schedule ──────────────────────────────────
        pmmt_schedule=SCHEDULE,
    )
