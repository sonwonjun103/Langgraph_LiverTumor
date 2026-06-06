import datetime
import json
import os
import subprocess
import argparse
import pandas as pd

from config import *
from logger import setup_logger
from train.trainer import Trainer


def format_path_part(value):
    if pd.isna(value):
        return "0"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def get_phase_filenames(data_type):
    if data_type == "ct":
        return "A.nii.gz", "P.nii.gz", "D.nii.gz"
    if data_type == "liver":
        return "AliverAv.nii.gz", "PliverPv.nii.gz", "DliverDv.nii.gz"

    raise ValueError(f"Unknown data_type: {data_type}")


def build_paths(dataframe, data_root1, data_root2, data_type="liver"):
    A, P, D, label = [], [], [], []
    a_name, p_name, d_name = get_phase_filenames(data_type)

    for _, row in dataframe.iterrows():
        subject = format_path_part(row["subject"])
        date = format_path_part(row["date"])

        if date == "0":
            case_dir = os.path.join(data_root2, subject)
        else:
            case_dir = os.path.join(data_root1, subject, date)

        A.append(os.path.join(case_dir, a_name))
        P.append(os.path.join(case_dir, p_name))
        D.append(os.path.join(case_dir, d_name))
        label.append(os.path.join(case_dir, "label.nii.gz"))

    return A, P, D, label


def filter_existing_cases(A, P, D, label, logger=None):
    kept = []
    missing = 0

    for paths in zip(A, P, D, label):
        if all(os.path.exists(path) for path in paths):
            kept.append(paths)
        else:
            missing += 1

    if missing:
        message = f"Skipped {missing} cases because one or more files were missing."
        if logger is not None:
            logger.info(message)
        else:
            print(message)

    if not kept:
        return [], [], [], []

    kept_A, kept_P, kept_D, kept_label = zip(*kept)
    return list(kept_A), list(kept_P), list(kept_D), list(kept_label)


def get_model(args):
    if args.model == "unet":
        from train.models.UNet import build_unet

        return build_unet(args)

    if args.model == "swinunetr":
        from train.models.SwinUNetR import build_swinunetr

        return build_swinunetr(args)

    if args.model == "sam_adapter":
        from train.models.SAMAdapter import build_sam_adapter

        return build_sam_adapter(args)

    raise ValueError(f"Unknown model: {args.model}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["unet", "swinunetr", "sam_adapter", "nnunetv2"],
        default="unet",
        help="Baseline model for 3-phase liver tumor segmentation.",
    )
    parser.add_argument("--data_root1", default=DATA_PATH1)
    parser.add_argument("--data_root2", default=DATA_PATH2)
    parser.add_argument(
        "--data_type",
        choices=["ct", "liver"],
        default="liver",
        help="Use original CT phases or liver-cropped phases.",
    )
    parser.add_argument("--sweep_xlsx", default="./register_dice_sweep.xlsx",
                        help="register_dice_sweep output. Top test_size cases (mean1 ascending) "
                             "become the test set; the rest are train.")
    parser.add_argument("--test_size", type=int, default=34)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--window", type=float, nargs=2, default=(0, 150))
    parser.add_argument(
        "--detection_thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="Lesion-level IoU thresholds for validation detection metrics.",
    )
    parser.add_argument(
        "--roi_size",
        type=int,
        nargs=3,
        default=[96, 128, 128],
        help="Patch/sliding-window size in SimpleITK array order: D H W.",
    )
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--pos_ratio", type=float, default=0.8,
                        help="Pos:neg ratio for RandCropByPosNegLabeld (only used when --random_crop).")
    parser.add_argument("--random_crop", action=argparse.BooleanOptionalAction, default=True,
                        help="Use RandCropByPosNegLabeld for training patches. "
                             "Default True. Use --no-random_crop to disable (full-volume training).")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply geometric + intensity augmentation (flip/rot, affine, noise, "
                             "blur, intensity scale/shift, gamma).")
    parser.add_argument("--use_sliding_window", action="store_true", default=True)
    parser.add_argument("--no_sliding_window", dest="use_sliding_window", action="store_false")
    parser.add_argument("--sw_batch_size", type=int, default=4)
    parser.add_argument("--sw_overlap", type=float, default=0.5)
    parser.add_argument("--sw_mode", choices=["constant", "gaussian"], default="gaussian")

    # 3D UNet arguments
    parser.add_argument("--channels", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 2, 2, 2])
    parser.add_argument("--num_res_units", type=int, default=2)

    # SwinUNETR arguments. MONAI >= 1.5 no longer uses img_size for SwinUNETR.
    parser.add_argument("--img_size", type=int, nargs=3, default=[96, 384, 384])
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--use_checkpoint", action="store_true")

    # 3DSAMAdapter arguments. This model uses the official encoder/prompt/decoder
    # training path instead of the generic Trainer.forward(image) loop.
    parser.add_argument("--sam_checkpoint", default="./train/models/3DSAMAdapter/ckpt/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam_pretrained_ckpt", default="./train/models/3DSAMAdapter/lits/last.pth.tar")
    parser.add_argument("--sam_eval_interval", type=int, default=None)
    parser.add_argument("--sam_num_pos_points", type=int, default=10)
    parser.add_argument("--sam_num_neg_points", type=int, default=20)
    parser.add_argument("--sam_val_num_neg_points", type=int, default=10)
    parser.add_argument("--sam_dice_weight", type=float, default=0.5)
    parser.add_argument("--sam_ce_weight", type=float, default=0.5)

    # nnUNetv2 arguments. Used only when --model nnunetv2. nnUNetv2 manages its
    # own data layout via the nnUNet_raw/preprocessed/results env vars, so
    # alldata_metrics.xlsx is not consulted in this mode.
    parser.add_argument("--nnunet_dataset", default="001",
                        help="Dataset id passed to nnUNetv2_train (e.g. 001, 002).")
    parser.add_argument("--nnunet_config", default="3d_fullres",
                        help="nnUNetv2 configuration (3d_fullres, 3d_lowres, 2d, ...).")
    parser.add_argument("--nnunet_folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Folds to train sequentially.")
    parser.add_argument("--nnunet_extra_args", nargs=argparse.REMAINDER, default=[],
                        help="Extra args forwarded to nnUNetv2_train (e.g. --c to resume). "
                             "Must come last on the command line.")
    parser.add_argument("--nnunet_raw",
                        default="/home/sonwonjun/research/Liver/finalcode/nnUNet/nnUNet_raw")
    parser.add_argument("--nnunet_preprocessed",
                        default="/home/sonwonjun/research/Liver/finalcode/nnUNet/nnUNet_preprocessed")
    parser.add_argument("--nnunet_results",
                        default="/home/sonwonjun/research/Liver/finalcode/nnUNet/nnUNet_results")
    parser.add_argument("--nnunet_log_dir", default=None,
                        help="Directory for per-fold logs. Defaults to <output_dir>/logs.")

    parser.add_argument("--output_dir", default="./checkpoints")
    args = parser.parse_args()
    # Layer the output directory by model, data_type, and a per-run timestamp
    # so every training run is isolated: checkpoints, training_curves.png,
    # training_history.json, the log file, and run_metadata.json all live
    # under the same folder.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.timestamp = timestamp
    args.output_dir = os.path.join(args.output_dir, args.model, args.data_type, timestamp)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, name=f"train_{args.model}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Output dir: {args.output_dir}")

    if args.model == "nnunetv2":
        from train.nnunetv2_runner import train_nnunetv2_all_folds

        train_nnunetv2_all_folds(args=args, logger=logger)
        return

    # Train/test split is driven by register_dice_sweep.xlsx (mean1 ascending) so
    # the worst-aligned cases form the held-out test set.
    from dataset_split import split_train_test

    train_data, test_data = split_train_test(
        sweep_xlsx=args.sweep_xlsx,
        test_size=args.test_size,
    )
    print(f"train={len(train_data)} cases, test={len(test_data)} cases (sweep={args.sweep_xlsx})")

    train_A, train_P, train_D, train_label = build_paths(
        train_data,
        args.data_root1,
        args.data_root2,
        data_type=args.data_type,
    )
    test_A, test_P, test_D, test_label = build_paths(
        test_data,
        args.data_root1,
        args.data_root2,
        data_type=args.data_type,
    )

    train_A, train_P, train_D, train_label = filter_existing_cases(
        train_A, train_P, train_D, train_label, logger=logger
    )
    test_A, test_P, test_D, test_label = filter_existing_cases(
        test_A, test_P, test_D, test_label, logger=logger
    )

    logger.info(f"Train cases: {len(train_A)}")
    logger.info(f"Validation cases: {len(test_A)}")
    if not train_A:
        raise RuntimeError(
            "No training cases were found. Check --data_root and the paths in --sweep_xlsx."
        )

    # Save a single, self-describing run_metadata.json under the timestamped
    # output_dir so any future ./Results/<timestamp>/ folder can be traced
    # back to the exact training config.
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_commit = None

    run_metadata = {
        "timestamp": timestamp,
        "model": args.model,
        "data_type": args.data_type,
        "git_commit": git_commit,
        "args": vars(args),
        "data": {
            "train_cases": len(train_A),
            "test_cases": len(test_A),
            "sweep_xlsx": args.sweep_xlsx,
        },
        "augmentation_pipeline": (
            [
                "RandFlipd(x3)", "RandRotate90d", "RandAffined",
                "RandGaussianNoised", "RandGaussianSmoothd",
                "RandScaleIntensityd", "RandShiftIntensityd",
                "RandAdjustContrastd",
            ]
            if getattr(args, "augment", False) else []
        ),
        "random_crop": getattr(args, "random_crop", True),
        "pos_ratio": getattr(args, "pos_ratio", 0.8),
    }
    with open(os.path.join(args.output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"Saved run_metadata.json under {args.output_dir}")

    if args.model == "sam_adapter":
        if args.roi_size != [128, 128, 128]:
            logger.info(
                "3DSAMAdapter reference training expects cubic crops; overriding roi_size to [128, 128, 128]."
            )
            args.roi_size = [128, 128, 128]
        if args.sam_eval_interval is None:
            args.sam_eval_interval = args.val_interval

        from train.sam_adapter_trainer import train_sam_adapter

        train_sam_adapter(
            args=args,
            train_paths=(train_A, train_P, train_D, train_label),
            val_paths=(test_A, test_P, test_D, test_label) if test_A else None,
            logger=logger,
        )
        return

    model = get_model(args)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model: {args.model}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    trainer = Trainer(
        args=args,
        model=model,
        A=train_A,
        P=train_P,
        D=train_D,
        label=train_label,
        val_A=test_A or None,
        val_P=test_P or None,
        val_D=test_D or None,
        val_label=test_label or None,
        logger=logger,
    )
    trainer.train()

if __name__=='__main__':
    main()
