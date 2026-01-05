# BRISC2025 — Segmentation + Classification

This repository contains the cleaned, end-to-end pipeline we used for the BRISC2025 course project:

- **Stage-1**: Binary tumor **segmentation** (U-Net or Attention U-Net)
- **Stage-2 (separate)**: Tumor-type **classification** using the **frozen Stage-1 encoder**
- **Joint (optional)**: **Multi-task U-Net** training (segmentation + classification together)
- **Classifier backbone comparison (optional)**: MobileNetV2 vs EfficientNet-B0 vs DenseNet121

## Dataset layout
This code expects the BRISC dataset root to contain:

```
<DATA_ROOT>/
  segmentation_task/
    train/
      images/
      masks/
    test/
      images/
      masks/
  classification_task/
    train/
      glioma/
      meningioma/
      pituitary/
      no_tumor/
    test/
      glioma/
      meningioma/
      pituitary/
      no_tumor/
```

> In Kaggle, the default root is typically `/kaggle/input/brisc2025/brisc2025`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

pip install -U pip
pip install -r requirements.txt
pip install -e .
```

> If you run into issues installing **torch/torchvision**, install them following the official PyTorch instructions for your OS/CUDA.

## Environment variables (optional)
You can set these instead of passing `--data-root` / `--out-dir` every time:

```bash
export BRISC_DATA_ROOT=/path/to/brisc2025
export BRISC_OUT_DIR=./outputs
```

## Training

### 1) Stage-1 Segmentation

U-Net:
```bash
python scripts/train_stage1.py --model unet --data-root "$BRISC_DATA_ROOT" --out-dir "$BRISC_OUT_DIR"
```

Attention U-Net:
```bash
python scripts/train_stage1.py --model attunet --data-root "$BRISC_DATA_ROOT" --out-dir "$BRISC_OUT_DIR"
```

Outputs:
- checkpoint: `outputs/stage1_<model>_seg_best.pth`
- loss/dice curves (shown as plots)
- summary table: train/val/test with **loss, mIoU, Dice, pixel accuracy**

### 2) Stage-2 Classification (separate)
Requires a Stage-1 checkpoint.

```bash
python scripts/train_stage2.py \
  --data-root "$BRISC_DATA_ROOT" \
  --out-dir "$BRISC_OUT_DIR" \
  --stage1-ckpt "$BRISC_OUT_DIR/stage1_unet_seg_best.pth"
```

Outputs:
- checkpoint: `outputs/stage2_cls_best.pth`
- summary table: train/val/test with **loss, accuracy, precision, recall, F1**

### 3) Joint multi-task training (optional)
```bash
python scripts/train_joint.py --data-root "$BRISC_DATA_ROOT" --out-dir "$BRISC_OUT_DIR"
```

Outputs:
- checkpoint: `outputs/joint_multitask_best.pth`
- segmentation + classification summaries for train/val/test

### 4) Compare 3 pretrained backbones (optional)
```bash
python scripts/compare_backbones.py --data-root "$BRISC_DATA_ROOT" --out-dir "$BRISC_OUT_DIR"
```

Outputs:
- comparison table (test metrics)
- loss + F1 curves

## Inference (demo-ready)

### Separate models (Stage-1 seg + Stage-2 cls)
- If `--image` is omitted, it picks a **random** image from `segmentation_task/test/images`.
- If `--mask` is omitted, it tries to auto-find the GT mask (if available) for IoU/Dice display.

```bash
python scripts/infer_separate.py \
  --data-root "$BRISC_DATA_ROOT" \
  --seg-model unet \
  --seg-ckpt "$BRISC_OUT_DIR/stage1_unet_seg_best.pth" \
  --cls-ckpt "$BRISC_OUT_DIR/stage2_cls_best.pth"
```

Manual image:
```bash
python scripts/infer_separate.py \
  --data-root "$BRISC_DATA_ROOT" \
  --seg-model unet \
  --seg-ckpt "$BRISC_OUT_DIR/stage1_unet_seg_best.pth" \
  --cls-ckpt "$BRISC_OUT_DIR/stage2_cls_best.pth" \
  --image "/path/to/demo_image.jpg"
```

### Joint model
```bash
python scripts/infer_joint.py --data-root "$BRISC_DATA_ROOT" --ckpt "$BRISC_OUT_DIR/joint_multitask_best.pth"
```

## Notes
- Images are processed as **grayscale (1-channel)** for U-Net based models.
- For pretrained ImageNet backbones we convert images to **RGB (3-channel)** and use ImageNet normalization.

