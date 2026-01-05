import argparse
import os

from brisc2025.pipeline import Paths, run_stage2_classification


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage-2: train classifier head on frozen Stage-1 encoder (separate training)"
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
    ap.add_argument(
        "--stage1-ckpt",
        required=True,
        help="Path to Stage-1 segmentation checkpoint (.pth)",
    )
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt-name", default="stage2_cls_best.pth")

    args = ap.parse_args()
    paths = Paths(data_root=args.data_root, out_dir=args.out_dir)

    run_stage2_classification(
        paths=paths,
        stage1_ckpt_path=args.stage1_ckpt,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        ckpt_name=args.ckpt_name,
    )


if __name__ == "__main__":
    main()
