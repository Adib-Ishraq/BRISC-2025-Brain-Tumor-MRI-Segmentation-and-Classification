import argparse
import os

from brisc2025.pipeline import Paths, run_backbone_compare


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare 3 ImageNet-pretrained classifier backbones (MobileNetV2/EfficientNetB0/DenseNet121)"
    )
    ap.add_argument(
        "--data-root",
        default=os.environ.get("BRISC_DATA_ROOT", "./data/brisc2025"),
        help="Dataset root containing segmentation_task/ and classification_task/",
    )
    ap.add_argument(
        "--out-dir",
        default=os.environ.get("BRISC_OUT_DIR", "./outputs"),
        help="Output directory",
    )
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)

    args = ap.parse_args()
    paths = Paths(data_root=args.data_root, out_dir=args.out_dir)

    run_backbone_compare(
        paths=paths,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        early_stop_patience=args.patience,
    )


if __name__ == "__main__":
    main()
