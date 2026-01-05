import argparse
import os

from brisc2025.pipeline import Paths, run_stage1_segmentation


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage-1: train binary tumor segmentation (U-Net / Attention U-Net)"
    )
    ap.add_argument(
        "--data-root",
        default=os.environ.get("BRISC_DATA_ROOT", "./data/brisc2025"),
        help="Dataset root containing segmentation_task/ and classification_task/",
    )
    ap.add_argument(
        "--out-dir",
        default=os.environ.get("BRISC_OUT_DIR", "./outputs"),
        help="Output directory for checkpoints and plots",
    )
    ap.add_argument("--model", choices=["unet", "attunet"], default="unet")
    ap.add_argument("--img-size", type=int, default=384, help="Square image size (e.g., 384 => 384x384)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    ap.add_argument(
        "--ckpt-name",
        default=None,
        help="Checkpoint filename (default auto: stage1_<model>_seg_best.pth)",
    )

    args = ap.parse_args()

    ckpt_name = args.ckpt_name
    if ckpt_name is None:
        ckpt_name = f"stage1_{args.model}_seg_best.pth"

    paths = Paths(data_root=args.data_root, out_dir=args.out_dir)

    run_stage1_segmentation(
        paths=paths,
        model_type=args.model,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        early_stop_patience=args.patience,
        ckpt_name=ckpt_name,
    )


if __name__ == "__main__":
    main()
