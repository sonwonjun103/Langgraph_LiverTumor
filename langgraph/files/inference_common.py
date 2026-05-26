"""Shared helpers for per-model tumor inference."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import SimpleITK as sitk

from train.data.dataset import normalize_volume


def resolve_phase_paths(attempt_dir, cfg) -> Tuple[Path, Path, Path]:
    """Return (A, P, D) NIfTI paths based on ``cfg.data_type``.

    - ``data_type == "ct"`` (default): use the registered originals
      ``attempt_dir/{A,P,D}.nii.gz``.
    - ``data_type == "liver"``: use the liver-masked versions
      ``attempt_dir/liver_images/{A,P,D}_liver.nii.gz`` produced by
      ``LiverExtractor`` during the registration / liver-gate step.

    Raises FileNotFoundError when the liver-masked files are requested but
    missing (likely because the fallback liver extractor was used instead of
    LiverExtractor).
    """
    attempt_dir = Path(attempt_dir)
    data_type = getattr(cfg, "data_type", "ct")
    if data_type == "liver":
        liver_dir = attempt_dir / "liver_images"
        paths = (
            liver_dir / "A_liver.nii.gz",
            liver_dir / "P_liver.nii.gz",
            liver_dir / "D_liver.nii.gz",
        )
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "cfg.data_type='liver' requires LiverExtractor outputs, but these "
                f"files are missing: {missing}. Make sure liver extraction ran "
                "(LiverExtractor) and produced the liver_images/ folder, or switch "
                "cfg.data_type back to 'ct'."
            )
        return paths
    return (
        attempt_dir / "A.nii.gz",
        attempt_dir / "P.nii.gz",
        attempt_dir / "D.nii.gz",
    )


def read_volume(path):
    image = sitk.ReadImage(str(path))
    volume = sitk.GetArrayFromImage(image)
    return image, volume


def load_case(a_path, p_path, d_path, window):
    reference_image, a_volume = read_volume(a_path)
    _, p_volume = read_volume(p_path)
    _, d_volume = read_volume(d_path)

    a_volume = normalize_volume(a_volume, window)
    p_volume = normalize_volume(p_volume, window)
    d_volume = normalize_volume(d_volume, window)

    image = np.stack([a_volume, p_volume, d_volume], axis=0)
    image = torch.from_numpy(image).unsqueeze(0).float()
    return image, reference_image


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(cleaned)
    return checkpoint


def save_nifti(array, reference_image, output_path):
    output_image = sitk.GetImageFromArray(array)
    output_image.CopyInformation(reference_image)
    sitk.WriteImage(output_image, str(output_path))


@torch.no_grad()
def sliding_window_predict(model, image, roi_size, sw_batch_size, overlap, mode):
    from monai.inferers import sliding_window_inference

    return sliding_window_inference(
        inputs=image,
        roi_size=tuple(roi_size),
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
        mode=mode,
    )
