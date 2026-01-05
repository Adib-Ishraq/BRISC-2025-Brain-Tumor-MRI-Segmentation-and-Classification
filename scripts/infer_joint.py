import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from brisc2025.pipeline import DEVICE, MultiTaskUNet, ID_TO_LABEL


def find_gt_mask(image_path: str, data_root: str) -> str | None:
    base = os.path.splitext(os.path.basename(image_path))[0] + ".png"
    candidates = [
        os.path.join(data_root, "segmentation_task", "train", "masks", base),
        os.path.join(data_root, "segmentation_task", "test", "masks", base),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def pick_random_test_image(data_root: str) -> str:
    img_dir = os.path.join(data_root, "segmentation_task", "test", "images")
    files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        raise FileNotFoundError(f"No images found in: {img_dir}")
    return os.path.join(img_dir, random.choice(files))


def load_gray_tensor(image_path: str, img_size: tuple[int, int]):
    img_orig = Image.open(image_path).convert("L")
    img_res = TF.resize(img_orig, img_size, interpolation=InterpolationMode.BILINEAR)
    x = TF.to_tensor(img_res).unsqueeze(0)
    return img_orig, img_res, x


def load_mask_bin(mask_path: str, img_size: tuple[int, int]) -> np.ndarray:
    m = Image.open(mask_path).convert("L")
    m = TF.resize(m, img_size, interpolation=InterpolationMode.NEAREST)
    return (np.array(m) > 0).astype(np.float32)


def seg_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7):
    p = pred.reshape(-1)
    t = gt.reshape(-1)
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    tn = ((1 - p) * (1 - t)).sum()

    iou_fg = (tp + eps) / (tp + fp + fn + eps)
    iou_bg = (tn + eps) / (tn + fn + fp + eps)
    miou = 0.5 * (iou_fg + iou_bg)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    pix = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    return float(miou), float(dice), float(pix)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inference for joint multitask model (seg + cls)")
    ap.add_argument("--data-root", default=os.environ.get("BRISC_DATA_ROOT", "./data/brisc2025"))
    ap.add_argument("--ckpt", required=True, help="Joint checkpoint (.pth)")
    ap.add_argument("--image", default="", help="Image path. If empty, picks random test image.")
    ap.add_argument("--mask", default="", help="Optional GT mask path. If empty, tries to auto-find.")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--thr", type=float, default=0.5)

    args = ap.parse_args()

    image_path = args.image.strip() or pick_random_test_image(args.data_root)
    mask_path = args.mask.strip() or find_gt_mask(image_path, args.data_root)

    print("DEVICE:", DEVICE)
    print("Image:", image_path)
    print("GT mask:", mask_path if mask_path else "(not found)")

    model = MultiTaskUNet(in_ch=1, seg_out_ch=1, num_classes=4, base=32).to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    model.eval()

    img_orig, img_proc, x = load_gray_tensor(image_path, (args.img_size, args.img_size))
    x = x.to(DEVICE)

    with torch.no_grad():
        seg_logits, cls_logits = model(x)
        seg_prob = torch.sigmoid(seg_logits)[0, 0].detach().cpu().numpy()
        pred_mask = (seg_prob > args.thr).astype(np.float32)
        pred_id = int(cls_logits.argmax(dim=1).item())
        pred_label = ID_TO_LABEL[pred_id]

    gt = None
    miou = dice = pix = None
    if mask_path and os.path.exists(mask_path):
        gt = load_mask_bin(mask_path, (args.img_size, args.img_size))
        miou, dice, pix = seg_metrics(pred_mask, gt)

    img_np = np.array(img_proc)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_orig, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(img_np, cmap="gray")
    if gt is not None:
        axes[1].imshow(gt, cmap="jet", alpha=0.4)
        axes[1].set_title("GT overlay")
    else:
        axes[1].set_title("GT overlay (N/A)")
    axes[1].axis("off")

    axes[2].imshow(img_np, cmap="gray")
    axes[2].imshow(pred_mask, cmap="jet", alpha=0.4)
    axes[2].set_title("Pred overlay")
    axes[2].axis("off")

    miou_txt = "N/A" if miou is None else f"{miou:.3f}"
    dice_txt = "N/A" if dice is None else f"{dice:.3f}"
    pix_txt = "N/A" if pix is None else f"{pix:.3f}"

    fig.suptitle(f"Pred class: {pred_label} | mIoU={miou_txt} Dice={dice_txt} PixAcc={pix_txt}", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
