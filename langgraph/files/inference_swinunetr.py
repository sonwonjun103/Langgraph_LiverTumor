"""SwinUNETR tumor inference for the pipeline."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from train.models.SwinUNetR import build_swinunetr

from .inference_common import (
    load_case,
    load_checkpoint,
    resolve_phase_paths,
    save_nifti,
    sliding_window_predict,
)


@torch.no_grad()
def run_swinunetr(attempt_dir: Path, output_dir: Path, cfg) -> Optional[str]:
    checkpoint_path = cfg.checkpoints.get("swinunetr")
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print(f"skip swinunetr: checkpoint not found -> {checkpoint_path}")
        return None

    device = torch.device(cfg.device)
    args = SimpleNamespace(
        img_size=tuple(cfg.img_size),
        feature_size=cfg.feature_size,
        use_checkpoint=cfg.use_checkpoint,
    )
    model = build_swinunetr(args).to(device)
    model.eval()
    load_checkpoint(model, checkpoint_path, device)

    a_path, p_path, d_path = resolve_phase_paths(attempt_dir, cfg)
    image, reference_image = load_case(a_path, p_path, d_path, cfg.window)
    image = image.to(device)
    logits = sliding_window_predict(
        model, image, cfg.roi_size, cfg.sw_batch_size, cfg.sw_overlap, cfg.sw_mode
    )
    probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (probability >= cfg.threshold).astype(np.uint8)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "pred.nii.gz"
    save_nifti(mask, reference_image, pred_path)
    return str(pred_path)
