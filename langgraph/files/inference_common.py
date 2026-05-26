"""Shared helpers for per-model tumor inference."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import SimpleITK as sitk

from train.data.dataset import normalize_volume


def resolve_phase_paths(attempt_dir, cfg) -> Tuple[Path, Path, Path]:
    """Return (A, P, D) NIfTI paths under ``attempt_dir``.

    The pipeline picks the right source file (CT vs liver-cropped) at the
    case-loading step (see ``get_case_from_row``) and copies it into
    ``input_dir/{A,P,D}.nii.gz``. After that everything downstream just reads
    those fixed filenames, regardless of cfg.data_type.
    """
    attempt_dir = Path(attempt_dir)
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
