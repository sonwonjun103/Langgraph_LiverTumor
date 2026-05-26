"""Sweep registration attempts 1..5 across qualifying cases and write only an
Excel table — no intermediate files, no langgraph imports.

For every case in data1metrics.xlsx (content blank + tumor_size >= threshold)
and data2metrics.xlsx (tumor_size >= threshold), this script:

  1. Reads the raw A/P/D NIfTI volumes from disk.
  2. For attempt N = 1..5, runs registration with REGISTRATION_CONFIGS[N]
     entirely in memory, dumps the registered phases into a per-case
     tempdir, runs TotalSegmentator (extract_segmentation) on each phase,
     and computes the mean of (AP, AD, PD) liver Dice.
  3. If mean Dice >= --liver_dice_threshold, stops early and fills the
     remaining attempts with 0.0. Otherwise moves to attempt N+1.
  4. The tempdir is deleted after each case so nothing is left on disk.

Output Excel columns:
  subject | date | tumor_size | mean1 | mean2 | mean3 | mean4 | mean5 | elapsed_min

Uses the SimpleITK Registration, TotalSegmentator wrapper, and compute_dice
function defined in ablationregis.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

from ablationregis import (
    REGISTRATION_CONFIGS,
    Registration,
    compute_dice,
    extract_segmentation,
)


DATA_ROOT_DEFAULT = "/mnt/d/research/Liver"


def _format_id(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def mean_liver_dice(rA: sitk.Image, rP: sitk.Image, rD: sitk.Image) -> float:
    """Save registered phases to a tempdir, run TotalSegmentator per phase, and
    return the mean of (AP, AD, PD) liver Dice. Tempdir is removed on exit."""
    with tempfile.TemporaryDirectory(prefix="reg_dice_") as tmp_str:
        tmp = Path(tmp_str)
        for phase, img in zip(("A", "P", "D"), (rA, rP, rD)):
            sitk.WriteImage(img, str(tmp / f"{phase}.nii.gz"))

        masks = []
        for phase in ("A", "P", "D"):
            mask_dir = tmp / f"{phase}_seg"
            mask_dir.mkdir(parents=True, exist_ok=True)
            extract_segmentation(str(tmp / f"{phase}.nii.gz"), str(mask_dir))
            mask_path = mask_dir / "liver.nii.gz"
            masks.append(sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))))

    a, p, d = masks
    ap = compute_dice(a, p)
    ad = compute_dice(a, d)
    pd_ = compute_dice(p, d)
    return float((ap + ad + pd_) / 3.0)


def sweep_case(case_dir: Path, threshold: float, max_attempts: int = 5) -> Tuple[List[float], float]:
    """Return (means, elapsed_seconds) for one case.

    means[i-1] = mean liver Dice after attempt i.
    Once an attempt passes the threshold, the rest are filled with 0.0.
    """
    rA_path = str(case_dir / "A.nii.gz")
    rP_path = str(case_dir / "P.nii.gz")
    rD_path = str(case_dir / "D.nii.gz")

    means: List[float] = []
    passed = False
    start = time.time()

    for attempt in range(1, max_attempts + 1):
        if passed:
            means.append(0.0)
            continue
        register = Registration(REGISTRATION_CONFIGS[attempt], attempt)
        rA, rP, rD = register.run(rA_path, rP_path, rD_path)
        m = mean_liver_dice(rA, rP, rD)
        means.append(m)
        if m >= threshold:
            passed = True

    elapsed = time.time() - start
    return means, elapsed


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data1_xlsx", default="./files/data1metrics.xlsx")
    parser.add_argument("--data2_xlsx", default="./files/data2metrics.xlsx")
    parser.add_argument("--data1_root", default=os.path.join(DATA_ROOT_DEFAULT, "data1correct"))
    parser.add_argument("--data2_root", default=os.path.join(DATA_ROOT_DEFAULT, "data2"))
    parser.add_argument("--tumor_size_threshold", type=float, default=2000.0)
    parser.add_argument("--liver_dice_threshold", type=float, default=0.95)
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--output", default="./register_dice_sweep.xlsx")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args)
    print(f"Loaded {len(cases)} cases (tumor_size >= {args.tumor_size_threshold})")

    rows = []
    total_start = time.time()
    for idx, case in enumerate(cases, start=1):
        print(f"\n[{idx}/{len(cases)}] {case['case_id']}  tumor_size={case['tumor_size']:.0f}")
        means, elapsed = sweep_case(
            case_dir=case["case_dir"],
            threshold=args.liver_dice_threshold,
            max_attempts=args.max_attempts,
        )
        elapsed_min = elapsed / 60.0
        for k, v in enumerate(means, start=1):
            print(f"   mean{k} = {v:.4f}")
        print(f"   elapsed: {elapsed_min:.2f} min ({elapsed:.1f}s)")

        rows.append({
            "subject": case["subject"],
            "date": case["date"],
            "tumor_size": case["tumor_size"],
            **{f"mean{i+1}": means[i] for i in range(len(means))},
            "elapsed_min": round(elapsed_min, 3),
        })

        # Save incrementally so partial runs can still be inspected.
        pd.DataFrame(rows).to_excel(args.output, index=False)

    total_min = (time.time() - total_start) / 60.0
    print(f"\nDone. {len(rows)} cases written to {args.output}  (total {total_min:.1f} min)")


if __name__ == "__main__":
    main()
