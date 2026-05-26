"""VoxelMorph-based deformable registration for the liver tumor pipeline.

Drop-in alternative to the SimpleITK iterative registration. The Portal
Venous phase is treated as the fixed image; the Arterial and Delayed phases
are warped to align with it using a pre-trained VoxelMorph network. This
matches the rest of the pipeline, where P is used as the reference frame
(see ``resample_label_to_reference`` in ``pipeline.ipynb``).

Inputs at ``input_folder``:
  - A.nii.gz, P.nii.gz, D.nii.gz

Outputs at ``output_path / f"voxelmorph_attempt_{attempt}"``:
  - A.nii.gz  (warped to P)
  - P.nii.gz  (copy)
  - D.nii.gz  (warped to P)
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk
import torch


def _read_image(path: Path):
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    return image, array


def _write_image_like(reference: sitk.Image, array: np.ndarray, path: Path) -> None:
    out = sitk.GetImageFromArray(array.astype(np.float32))
    out.CopyInformation(reference)
    sitk.WriteImage(out, str(path))


def _normalize(volume: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    low, high = window
    clipped = np.clip(volume, low, high)
    return (clipped - low) / max(high - low, 1e-6)


def _pad_to_multiple(volume: np.ndarray, multiple: int = 16):
    pads = []
    for size in volume.shape:
        remainder = size % multiple
        extra = 0 if remainder == 0 else multiple - remainder
        left = extra // 2
        right = extra - left
        pads.append((left, right))
    padded = np.pad(volume, pads, mode="constant", constant_values=0.0)
    return padded, pads


def _crop_to_original(volume: np.ndarray, pads):
    slices = tuple(slice(p[0], v - p[1]) for v, p in zip(volume.shape, pads))
    return volume[slices]


class VoxelMorphRegistration:
    """Apply a pre-trained VoxelMorph model to register P/D to A."""

    def __init__(
        self,
        input_folder: str,
        output_path: str,
        model_path: str,
        device: str = None,
        window: Tuple[float, float] = (-200.0, 300.0),
        attempt: int = 0,
    ):
        self.input_folder = Path(input_folder)
        self.output_path = Path(output_path)
        self.model_path = model_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.window = window
        self.attempt = attempt

    def _load_model(self):
        try:
            import voxelmorph as vxm
        except ImportError as exc:
            raise ImportError(
                "voxelmorph is not installed. Install with: pip install voxelmorph"
            ) from exc

        model = vxm.networks.VxmDense.load(self.model_path, self.device)
        model.to(self.device).eval()
        return model

    @torch.no_grad()
    def _register_pair(self, model, fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
        fixed_padded, pads = _pad_to_multiple(fixed)
        moving_padded, _ = _pad_to_multiple(moving)

        fixed_t = torch.from_numpy(fixed_padded[None, None]).float().to(self.device)
        moving_t = torch.from_numpy(moving_padded[None, None]).float().to(self.device)

        moved_t, _ = model(moving_t, fixed_t, registration=True)
        moved = moved_t[0, 0].cpu().numpy()
        return _crop_to_original(moved, pads)

    def run(self) -> str:
        attempt_dir = self.output_path / f"voxelmorph_attempt_{self.attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        _, a_array = _read_image(self.input_folder / "A.nii.gz")
        p_image, p_array = _read_image(self.input_folder / "P.nii.gz")
        _, d_array = _read_image(self.input_folder / "D.nii.gz")

        a_norm = _normalize(a_array, self.window)
        p_norm = _normalize(p_array, self.window)
        d_norm = _normalize(d_array, self.window)

        model = self._load_model()
        # P is the fixed/reference image; A and D are warped to align with P.
        moved_a_norm = self._register_pair(model, fixed=p_norm, moving=a_norm)
        moved_d_norm = self._register_pair(model, fixed=p_norm, moving=d_norm)

        low, high = self.window
        moved_a = moved_a_norm * (high - low) + low
        moved_d = moved_d_norm * (high - low) + low

        _write_image_like(p_image, moved_a, attempt_dir / "A.nii.gz")
        _write_image_like(p_image, p_array, attempt_dir / "P.nii.gz")
        _write_image_like(p_image, moved_d, attempt_dir / "D.nii.gz")

        return str(attempt_dir)
