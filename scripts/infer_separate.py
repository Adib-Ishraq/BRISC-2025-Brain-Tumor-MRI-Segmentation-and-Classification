import argparse
import os
import random

import torch

from brisc2025.pipeline import (
    DEVICE,
    Paths,
    UNet,
    AttentionUNet,
    UNetEncoder,
    EncoderClassifier,
    infer_separate,
)


def find_gt_mask(image_path: str, data_root: str) -> str | None:
    """Try to locate the GT mask (same base name, .png) in train/test mask folders."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Inference for separate training (Stage-1 seg + Stage-2 cls)")
    ap.add_argument("--data-root", default=os.environ.get("BRISC_DATA_ROOT", "./data/brisc2025"))
    ap.add_argument("--seg-ckpt", required=True, help="Stage-1 segmentation checkpoint (.pth)")
    ap.add_argument("--cls-ckpt", required=True, help="Stage-2 classification checkpoint (.pth)")
    ap.add_argument("--seg-model", choices=["unet", "attunet"], default="unet")
    ap.add_argument("--image", default="", help="Path to a demo image. If empty, picks random test image.")
    ap.add_argument("--mask", default="", help="Optional GT mask path. If empty, tries to auto-find.")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--thr", type=float, default=0.5)

    args = ap.parse_args()

    data_root = args.data_root
    image_path = args.image.strip() or pick_random_test_image(data_root)
    mask_path = args.mask.strip() or find_gt_mask(image_path, data_root)

    print("DEVICE:", DEVICE)
    print("Image:", image_path)
    print("GT mask:", mask_path if mask_path else "(not found)")

    # --- load seg model ---
    if args.seg_model == "unet":
        seg_model = UNet(in_ch=1, out_ch=1, base=32).to(DEVICE)
    else:
        seg_model = AttentionUNet(in_ch=1, out_ch=1, base=32).to(DEVICE)
    seg_model.load_state_dict(torch.load(args.seg_ckpt, map_location=DEVICE))

    # --- build classifier (encoder from Stage-1 ckpt) + load Stage-2 weights ---
    encoder = UNetEncoder(in_ch=1, base=32)
    state = torch.load(args.seg_ckpt, map_location="cpu")
    enc_state = {k: v for k, v in state.items() if k.startswith(("enc1", "enc2", "enc3", "enc4", "bottleneck"))}
    encoder.load_state_dict(enc_state, strict=False)
    for p in encoder.parameters():
        p.requires_grad = False

    cls_model = EncoderClassifier(encoder.to(DEVICE), feat_ch=32 * 16, num_classes=4).to(DEVICE)
    cls_model.load_state_dict(torch.load(args.cls_ckpt, map_location=DEVICE))

    infer_separate(
        seg_model=seg_model,
        cls_model=cls_model,
        image_path=image_path,
        mask_path=mask_path,
        img_size=(args.img_size, args.img_size),
        thr=args.thr,
    )


if __name__ == "__main__":
    main()
