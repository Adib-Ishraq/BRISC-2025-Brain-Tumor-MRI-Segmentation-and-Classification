import argparse
import os

from brisc2025.pipeline import Paths, run_joint_multitask


def main() -> None:
    ap = argparse.ArgumentParser(description="Joint multi-task training (segmentation + classification)")
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
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--lambda-cls",
        type=float,
        default=0.5,
        help="Classification loss weight in total loss: L = Lseg + lambda_cls * Lcls",
    )
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--ckpt-name", default="joint_multitask_best.pth")

    args = ap.parse_args()
    paths = Paths(data_root=args.data_root, out_dir=args.out_dir)

    run_joint_multitask(
        paths=paths,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lambda_cls=args.lambda_cls,
        early_stop_patience=args.patience,
        ckpt_name=args.ckpt_name,
    )


if __name__ == "__main__":
    main()
