"""nnUNetv2 tumor inference via CLI for the pipeline."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def run_nnunetv2(attempt_dir: Path, output_dir: Path, cfg, case_id: str) -> Optional[str]:
    attempt_dir = Path(attempt_dir)
    output_dir = Path(output_dir)
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(attempt_dir / "A.nii.gz", input_dir / f"{case_id}_0000.nii.gz")
    shutil.copy2(attempt_dir / "P.nii.gz", input_dir / f"{case_id}_0001.nii.gz")
    shutil.copy2(attempt_dir / "D.nii.gz", input_dir / f"{case_id}_0002.nii.gz")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if cfg.nnunet_raw:
        env["nnUNet_raw"] = cfg.nnunet_raw
    if cfg.nnunet_preprocessed:
        env["nnUNet_preprocessed"] = cfg.nnunet_preprocessed
    if cfg.nnunet_results:
        env["nnUNet_results"] = cfg.nnunet_results

    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", cfg.nnunet_dataset,
        "-c", cfg.nnunet_config,
    ]
    if cfg.nnunet_save_probabilities:
        cmd.append("--save_probabilities")

    log_path = output_dir / "nnunetv2_predict.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=env)

    pred_candidates = sorted(
        path for path in output_dir.glob("*.nii.gz")
        if path.name != "pred.nii.gz"
    )
    if process.returncode != 0 or not pred_candidates:
        print(f"nnUNetv2 failed. See {log_path}")
        return None

    pred_path = output_dir / "pred.nii.gz"
    shutil.copy2(pred_candidates[0], pred_path)
    return str(pred_path)
