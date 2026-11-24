"""MMSSR-Net training script.

MMSSR-Net uses Progressive Mixed-Modality Training (RG-PMMT) to learn
robust cross-modal representations that work with dual, transmissive-only,
and reflective-only inputs.
"""

from ultralytics import YOLO

# ─── User config ───────────────────────────────────────────────
MODEL_CFG = "ultralytics/cfg/models/fuse/DEYOLOseed12.yaml"    # MMSSR-Net model YAML (same layers as BiSSR-Net)
DATA_YAML = "ultralytics/cfg/datasets/obb-seed.yaml"            # Dataset YAML
SCHEDULE  = "full_warmup_120"   # four-stage schedule reported in the paper (Table S1):
                                # 1-120 dual-only 100/0/0 -> 121-180 90/5/5 -> 181-255 80/10/10 -> 256-300 95/2.5/2.5
                                # alternatives: "conservative" | "legacy"
IMG_SZ    = 1024
BATCH     = 32
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
        workers=0,  # original training setting
        device=DEVICE,
        optimizer="SGD",
        patience=40,
        amp=False,
        cache="disk",
        close_mosaic=0,  # keep mosaic augmentation until the last epoch (original training setting)
        project="runs/train_mmssr",
        name="mmssr_net",
        resume=False,
        fraction=1.0,
        # ─── RG-PMMT schedule ──────────────────────────────────
        pmmt_schedule=SCHEDULE,
    )
