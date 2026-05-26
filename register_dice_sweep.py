"""Sweep registration attempts 1..5 across qualifying cases and write the
per-attempt mean liver Dice to an Excel sheet.

For every case in data1metrics.xlsx (with content blank, tumor_size >= 2000)
and data2metrics.xlsx (tumor_size >= 2000):

  - Apply registration attempt 1, evaluate the liver gate (mean of AP/AD/PD
    Dice using TotalSegmentator masks).
  - If mean Dice >= --liver_dice_threshold, stop early and fill the remaining
    attempts with 0.0.
  - Otherwise move to attempt 2 and repeat, up to attempt 5.

Output Excel columns:
  subject | date | tumor_size | mean1 | mean2 | mean3 | mean4 | mean5

This is a cleaned-up replacement for the ablation logic embedded in
ablationregis.py — same idea, but reuses langgraph.files.registration and
langgraph.files.liver_extractor and writes a compact result table.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import SimpleITK as sitk

from langgraph.files.config import REGISTRATION_CONFIGS
from langgraph.files.registration import Registration
from langgraph.files.liver_extractor import LiverExtractor


DATA_ROOT_DEFAULT = "/mnt/d/research/Liver"


def _format_id(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 2.0 * inter / denom if denom > 0 else 1.0


def binary_array(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0


def mean_liver_dice(attempt_dir: Path) -> float:
    """Run LiverExtractor (TotalSegmentator) and return mean(AP, AD, PD) Dice."""
    extractor = LiverExtractor(input_folder=str(attempt_dir), output_path=str(attempt_dir))
    mask_paths = extractor.run_all()
    if not (isinstance(mask_paths, dict) and all(p in mask_paths for p in ("A", "P", "D"))):
        return 0.0
    a = binary_array(mask_paths["A"])
    p = binary_array(mask_paths["P"])
    d = binary_array(mask_paths["D"])
    ap = dice_score(a, p)
    ad = dice_score(a, d)
    pd_ = dice_score(p, d)
    return float((ap + ad + pd_) / 3.0)


def load_cases(args) -> List[dict]:
    data1 = pd.read_excel(args.data1_xlsx)
    data2 = pd.read_excel(args.data2_xlsx)

    if "content" in data1.columns:
        data1 = data1[data1["content"].isna() & (data1["tumor_size"] >= args.tumor_size_threshold)]
    else:
        data1 = data1[data1["tumor_size"] >= args.tumor_size_threshold]
    data2 = data2[data2["tumor_size"] >= args.tumor_size_threshold]

    cases: List[dict] = []

    for _, row in data1.iterrows():
        subject = _format_id(row["subject"])
        date = _format_id(row["date"])
        folder = Path(args.data1_root) / subject / date
        if all((folder / f"{p}.nii.gz").exists() for p in ("A", "P", "D")):
            cases.append({
                "subject": subject,
                "date": date,
                "tumor_size": float(row["tumor_size"]),
                "case_dir": folder,
                "case_id": f"{subject}_{date}",
            })
        else:
            print(f"[skip] missing inputs: {folder}", file=sys.stderr)

    for _, row in data2.iterrows():
        subject = _format_id(row["subject"])
        folder = Path(args.data2_root) / subject
        if all((folder / f"{p}.nii.gz").exists() for p in ("A", "P", "D")):
            cases.append({
                "subject": subject,
                "date": "",
                "tumor_size": float(row["tumor_size"]),
                "case_dir": folder,
                "case_id": subject,
            })
        else:
            print(f"[skip] missing inputs: {folder}", file=sys.stderr)

    return cases


def sweep_case(case_dir: Path, case_result_dir: Path, threshold: float, max_attempts: int) -> List[float]:
    """Return [mean1, mean2, ..., meanN]; 0.0 padding after the first pass."""
    means: List[float] = []
    passed = False
    for attempt in range(1, max_attempts + 1):
        if passed:
            means.append(0.0)
            continue
        reg_param = REGISTRATION_CONFIGS[attempt]
        attempt_dir = Path(Registration(
            regis_param=reg_param,
            input_folder=str(case_dir),
            output_path=str(case_result_dir),
            attempt=attempt - 1,
        ).run())
        m = mean_liver_dice(attempt_dir)
        means.append(m)
        if m >= threshold:
            passed = True
    return means


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data1_xlsx", default="./files/data1metrics.xlsx")
    parser.add_argument("--data2_xlsx", default="./files/data2metrics.xlsx")
    parser.add_argument("--data1_root", default=os.path.join(DATA_ROOT_DEFAULT, "data1correct"),
                        help="Root for data1 cases. Default matches ablationregis.py.")
    parser.add_argument("--data2_root", default=os.path.join(DATA_ROOT_DEFAULT, "data2"))
    parser.add_argument("--tumor_size_threshold", type=float, default=2000.0)
    parser.add_argument("--liver_dice_threshold", type=float, default=0.95)
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--results_root", default="./Results/register_dice_sweep",
                        help="Per-attempt registration outputs are written here.")
    parser.add_argument("--output", default="./register_dice_sweep.xlsx",
                        help="Output Excel file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args)
    print(f"Loaded {len(cases)} cases (tumor_size >= {args.tumor_size_threshold})")

    Path(args.results_root).mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, case in enumerate(cases, start=1):
        case_result_dir = Path(args.results_root) / case["case_id"]
        case_result_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx}/{len(cases)}] {case['case_id']}  tumor_size={case['tumor_size']:.0f}")
        means = sweep_case(
            case_dir=case["case_dir"],
            case_result_dir=case_result_dir,
            threshold=args.liver_dice_threshold,
            max_attempts=args.max_attempts,
        )
        for k, v in enumerate(means, start=1):
            print(f"   mean{k} = {v:.4f}")

        rows.append({
            "subject": case["subject"],
            "date": case["date"],
            "tumor_size": case["tumor_size"],
            **{f"mean{i+1}": means[i] for i in range(len(means))},
        })

        # Save incrementally so partial runs can still be inspected.
        pd.DataFrame(rows).to_excel(args.output, index=False)

    print(f"\nDone. {len(rows)} cases written to {args.output}")


if __name__ == "__main__":
    main()
