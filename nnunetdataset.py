import argparse
import json
import os
import shutil
from dataclasses import dataclass

import pandas as pd

from config import DATA_PATH1, DATA_PATH2


@dataclass
class DatasetConfig:
    folder: str
    name: str
    description: str
    phase_files: tuple[str, str, str]


DATASET_CONFIGS = {
    "Dataset001": DatasetConfig(
        folder="Dataset001",
        name="LiverTumorCT",
        description="Original 3-phase CT liver tumor segmentation",
        phase_files=("A.nii.gz", "P.nii.gz", "D.nii.gz"),
    ),
    "Dataset002": DatasetConfig(
        folder="Dataset002",
        name="LiverTumorLiverCrop",
        description="Liver-cropped 3-phase CT liver tumor segmentation",
        phase_files=("AliverAv.nii.gz", "PliverPv.nii.gz", "DliverDv.nii.gz"),
    ),
}


def format_path_part(value):
    if pd.isna(value):
        return "0"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def split_dataframe(sweep_xlsx, test_size):
    # Driven by register_dice_sweep.xlsx (mean1 ascending) so the worst-aligned
    # cases form the held-out test set; the rest are train.
    from dataset_split import split_train_test

    return split_train_test(sweep_xlsx=sweep_xlsx, test_size=test_size)


def get_case_info(row, data_root1, data_root2):
    subject = format_path_part(row["subject"])
    date = format_path_part(row["date"])

    if date == "0":
        source_dir = os.path.join(data_root2, subject)
        case_id = subject
        source_group = "data2"
    else:
        source_dir = os.path.join(data_root1, subject, date)
        case_id = f"{subject}_{date}"
        source_group = "data1"

    return {
        "subject": subject,
        "date": date,
        "source_dir": source_dir,
        "case_id": case_id,
        "source_group": source_group,
    }


def get_dataset_dirs(dataset_dir):
    return {
        "imagesTr": os.path.join(dataset_dir, "imagesTr"),
        "imagesTs": os.path.join(dataset_dir, "imagesTs"),
        "labelsTr": os.path.join(dataset_dir, "labelsTr"),
        "labelsTs": os.path.join(dataset_dir, "labelsTs"),
    }


def make_dataset_dirs(dataset_dir, dry_run=False):
    dirs = get_dataset_dirs(dataset_dir)
    if not dry_run:
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)
    return dirs


def build_case_file_pairs(row, config, split, dirs, data_root1, data_root2, channel_start):
    case = get_case_info(row, data_root1, data_root2)
    image_dir = dirs["imagesTr"] if split == "train" else dirs["imagesTs"]
    label_dir = dirs["labelsTr"] if split == "train" else dirs["labelsTs"]

    a_file, p_file, d_file = config.phase_files
    source_files = [
        os.path.join(case["source_dir"], a_file),
        os.path.join(case["source_dir"], p_file),
        os.path.join(case["source_dir"], d_file),
        os.path.join(case["source_dir"], "label.nii.gz"),
    ]
    target_files = [
        os.path.join(image_dir, f"{case['case_id']}_{channel_start:04d}.nii.gz"),
        os.path.join(image_dir, f"{case['case_id']}_{channel_start + 1:04d}.nii.gz"),
        os.path.join(image_dir, f"{case['case_id']}_{channel_start + 2:04d}.nii.gz"),
        os.path.join(label_dir, f"{case['case_id']}.nii.gz"),
    ]
    return case, list(zip(source_files, target_files))


def validate_sources(file_pairs):
    missing = [src for src, _ in file_pairs if not os.path.exists(src)]
    return missing


def copy_pairs(file_pairs, overwrite=False, dry_run=False):
    messages = []
    for src, dst in file_pairs:
        if os.path.exists(dst) and not overwrite:
            messages.append(f"exists: {dst}")
            continue

        if dry_run:
            messages.append(f"copy: {src} -> {dst}")
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        messages.append(f"copied: {dst}")

    return messages


def copy_case(row, config, split, dirs, args):
    case, file_pairs = build_case_file_pairs(
        row=row,
        config=config,
        split=split,
        dirs=dirs,
        data_root1=args.data_root1,
        data_root2=args.data_root2,
        channel_start=args.channel_start,
    )

    missing = validate_sources(file_pairs)
    if missing:
        return False, case, [f"missing: {path}" for path in missing]

    messages = copy_pairs(
        file_pairs,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    return True, case, messages


def write_dataset_json(dataset_dir, config, num_training, dry_run=False):
    dataset_json = {
        "channel_names": {
            "0": "Arterial",
            "1": "Portal",
            "2": "Delayed",
        },
        "labels": {
            "background": 0,
            "tumor": 1,
        },
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
        "name": config.name,
        "description": config.description,
        "reference": "",
        "licence": "",
        "release": "1.0",
    }

    json_path = os.path.join(dataset_dir, "dataset.json")
    if dry_run:
        return json_path

    os.makedirs(dataset_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(dataset_json, file, indent=4)
    return json_path


def create_dataset(config, train_data, test_data, args):
    dataset_dir = os.path.join(args.output_root, config.folder)
    dirs = make_dataset_dirs(dataset_dir, dry_run=args.dry_run)

    train_ok = 0
    test_ok = 0
    skipped = []

    for _, row in train_data.iterrows():
        ok, case, messages = copy_case(row, config, "train", dirs, args)
        if ok:
            train_ok += 1
        else:
            skipped.append((case["case_id"], messages))

    for _, row in test_data.iterrows():
        ok, case, messages = copy_case(row, config, "test", dirs, args)
        if ok:
            test_ok += 1
        else:
            skipped.append((case["case_id"], messages))

    json_path = write_dataset_json(
        dataset_dir=dataset_dir,
        config=config,
        num_training=train_ok,
        dry_run=args.dry_run,
    )

    print(f"\n{config.folder} ({config.name})")
    print(f"  output: {dataset_dir}")
    print(f"  train copied: {train_ok}")
    print(f"  test copied: {test_ok}")
    print(f"  dataset.json: {json_path}")

    if skipped:
        print(f"  skipped cases: {len(skipped)}")
        for case_id, messages in skipped[:20]:
            print(f"    {case_id}")
            for message in messages[:4]:
                print(f"      {message}")
        if len(skipped) > 20:
            print(f"    ... {len(skipped) - 20} more")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create nnUNet Dataset001/002 from the same split used by main.py."
    )
    parser.add_argument("--sweep_xlsx", default="./register_dice_sweep.xlsx",
                        help="register_dice_sweep output. Top test_size cases (mean1 ascending) "
                             "become the test set; the rest are train.")
    parser.add_argument("--data_root1", default=DATA_PATH1)
    parser.add_argument("--data_root2", default=DATA_PATH2)
    parser.add_argument("--output_root", default="./nnUNet/nnUNet_raw")
    parser.add_argument("--test_size", type=int, default=34)
    parser.add_argument(
        "--dataset",
        choices=["all", "Dataset001", "Dataset002"],
        default="all",
        help="Dataset001=original CT, Dataset002=liver-cropped CT.",
    )
    parser.add_argument(
        "--channel_start",
        type=int,
        default=0,
        help="nnUNet standard is 0 for _0000/_0001/_0002. Use 1 for _0001/_0002/_0003.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    train_data, test_data = split_dataframe(args.sweep_xlsx, args.test_size)

    if args.dataset == "all":
        selected_configs = DATASET_CONFIGS.values()
    else:
        selected_configs = [DATASET_CONFIGS[args.dataset]]

    print(f"Train rows: {len(train_data)}")
    print(f"Test rows: {len(test_data)}")
    print(f"Output root: {args.output_root}")
    print(f"Channel start: {args.channel_start}")
    if args.dry_run:
        print("Dry run: no files will be copied.")

    for config in selected_configs:
        create_dataset(config, train_data, test_data, args)


if __name__ == "__main__":
    main()
