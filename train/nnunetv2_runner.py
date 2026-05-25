"""Run nnUNetv2_train sequentially over a set of folds.

Invoked from main.py when ``args.model == "nnunetv2"``. nnUNetv2 uses its own
``nnUNet_raw``/``nnUNet_preprocessed``/``nnUNet_results`` directory layout, so
this runner does not use ``alldata_metrics.xlsx`` or the per-phase A/P/D
paths. Per-fold stdout/stderr is streamed to the console and also captured to
a log file.
"""
from __future__ import annotations

import os
import subprocess


def train_nnunetv2_all_folds(args, logger=None) -> None:
    dataset = getattr(args, "nnunet_dataset", "001")
    config = getattr(args, "nnunet_config", "3d_fullres")
    folds = getattr(args, "nnunet_folds", [0, 1, 2, 3, 4])
    extra = list(getattr(args, "nnunet_extra_args", []) or [])

    log_dir = getattr(args, "nnunet_log_dir", None) or os.path.join(
        getattr(args, "output_dir", "./checkpoints"), "logs"
    )
    os.makedirs(log_dir, exist_ok=True)

    env = os.environ.copy()
    for arg_key, env_key in (
        ("nnunet_raw", "nnUNet_raw"),
        ("nnunet_preprocessed", "nnUNet_preprocessed"),
        ("nnunet_results", "nnUNet_results"),
    ):
        value = getattr(args, arg_key, None)
        if value:
            env[env_key] = value

    def _emit(message: str) -> None:
        if logger is not None:
            logger.info(message)
        else:
            print(message)

    failed = []
    for fold in folds:
        log_path = os.path.join(log_dir, f"{dataset}_{config}_fold{fold}.log")
        _emit("=" * 60)
        _emit(f"Training nnUNetv2 Dataset={dataset} Config={config} Fold={fold}")
        _emit(f"Log file: {log_path}")
        if extra:
            _emit(f"Extra args: {' '.join(extra)}")
        _emit("=" * 60)

        cmd = ["nnUNetv2_train", str(dataset), str(config), str(fold), *extra]
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
            process.wait()

        if process.returncode != 0:
            failed.append(fold)
            _emit(f"Fold {fold} failed (return code {process.returncode}). See {log_path}")
        else:
            _emit(f"Fold {fold} completed.")

    _emit("=" * 60)
    if failed:
        raise RuntimeError(f"nnUNetv2 training failed for folds: {failed}")
    _emit(f"All folds finished for Dataset={dataset} Config={config}")
