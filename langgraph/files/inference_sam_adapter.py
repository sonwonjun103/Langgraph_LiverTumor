"""3D SAM Adapter tumor inference for the pipeline.

Adapted from the 3DSAM-adapter reference test script: builds the image encoder,
4 prompt encoders, and the mask decoder separately, then samples a point prompt
from the ground-truth label and runs a single patch centered on the prompt.
"""
from __future__ import annotations

import importlib
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .inference_common import load_case, save_nifti


def _load_adapter_modules():
    image_encoder_module = importlib.import_module("train.models.3DSAMAdapter.image_encoder")
    mask_decoder_module = importlib.import_module("train.models.3DSAMAdapter.mask_decoder")
    prompt_encoder_module = importlib.import_module("train.models.3DSAMAdapter.prompt_encoder")
    return (
        image_encoder_module.ImageEncoderViT_3d_v2,
        mask_decoder_module.VIT_MLAHead_h,
        prompt_encoder_module.PromptEncoder,
        prompt_encoder_module.TwoWayTransformer,
    )


def _build_models(checkpoint_path: str, device: torch.device):
    ImageEncoderViT_3d, VIT_MLAHead_h, PromptEncoder, TwoWayTransformer = _load_adapter_modules()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    img_encoder = ImageEncoderViT_3d(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        cubic_window_size=8,
        out_chans=256,
        num_slice=16,
    )
    img_encoder.load_state_dict(state["encoder_dict"], strict=True)
    img_encoder.to(device).eval()

    prompt_encoder_list = []
    for index in range(4):
        prompt_encoder = PromptEncoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            )
        )
        prompt_encoder.load_state_dict(state["feature_dict"][index], strict=True)
        prompt_encoder.to(device).eval()
        prompt_encoder_list.append(prompt_encoder)

    mask_decoder = VIT_MLAHead_h(img_size=96, num_classes=2)
    mask_decoder.load_state_dict(state["decoder_dict"], strict=True)
    mask_decoder.to(device).eval()

    return img_encoder, prompt_encoder_list, mask_decoder


def _read_gt_volume(gt_path: str) -> torch.Tensor:
    import SimpleITK as sitk

    array = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))
    return torch.from_numpy(array.astype(np.float32))


def _model_predict(
    img_patch: torch.Tensor,
    points_torch: torch.Tensor,
    img_encoder,
    prompt_encoder_list,
    mask_decoder,
    patch_size: int,
    device: torch.device,
) -> torch.Tensor:
    out = F.interpolate(img_patch.float(), scale_factor=512 / patch_size, mode="trilinear")
    input_batch = out[0].transpose(0, 1)
    batch_features, feature_list = img_encoder(input_batch)
    feature_list.append(batch_features)
    points_torch = points_torch.transpose(0, 1)

    new_feature = []
    for i, (feature, feature_decoder) in enumerate(zip(feature_list, prompt_encoder_list)):
        if i == 3:
            new_feature.append(
                feature_decoder(feature.to(device), points_torch.clone(), [patch_size, patch_size, patch_size])
            )
        else:
            new_feature.append(feature.to(device))

    img_resize = F.interpolate(
        img_patch[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device),
        scale_factor=64 / patch_size,
        mode="trilinear",
    )
    new_feature.append(img_resize)
    masks = mask_decoder(new_feature, 2, patch_size // 64)
    masks = masks.permute(0, 1, 4, 2, 3)
    return masks


@torch.no_grad()
def run_sam_adapter(
    attempt_dir: Path,
    output_dir: Path,
    cfg,
    gt_path: Optional[str] = None,
    num_prompts: int = 1,
    sam_roi_size: Tuple[int, int, int] = (128, 128, 128),
) -> Optional[str]:
    checkpoint_path = cfg.checkpoints.get("sam3d_adapter")
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print(f"skip sam3d_adapter: checkpoint not found -> {checkpoint_path}")
        return None
    if gt_path is None or not Path(gt_path).exists():
        print(f"skip sam3d_adapter: GT label required for prompt sampling -> {gt_path}")
        return None

    patch_size = sam_roi_size[0]
    if not (sam_roi_size[0] == sam_roi_size[1] == sam_roi_size[2]):
        print(f"skip sam3d_adapter: cubic patch required, got {sam_roi_size}")
        return None
    if patch_size % 64 != 0:
        print(f"skip sam3d_adapter: patch_size must be divisible by 64, got {patch_size}")
        return None

    device = torch.device(cfg.device)
    img_encoder, prompt_encoder_list, mask_decoder = _build_models(checkpoint_path, device)

    image, reference_image = load_case(
        attempt_dir / "A.nii.gz",
        attempt_dir / "P.nii.gz",
        attempt_dir / "D.nii.gz",
        cfg.window,
    )
    image = image.to(device)

    seg = _read_gt_volume(gt_path)
    seg = (seg > 0.5).float().unsqueeze(0).unsqueeze(0)
    prompt = F.interpolate(seg, image.shape[2:], mode="nearest")[0]

    positive = torch.where(prompt == 1)
    if positive[0].numel() == 0:
        print("skip sam3d_adapter: GT has no foreground voxels for prompt sampling")
        return None

    np.random.seed(0)
    sample = np.random.choice(np.arange(positive[0].numel()), num_prompts, replace=True)
    x = positive[1][sample].unsqueeze(1)
    y = positive[3][sample].unsqueeze(1)
    z = positive[2][sample].unsqueeze(1)

    x_m = (torch.max(x) + torch.min(x)) // 2
    y_m = (torch.max(y) + torch.min(y)) // 2
    z_m = (torch.max(z) + torch.min(z)) // 2

    d_min = (x_m - patch_size // 2).item()
    d_max = (x_m + patch_size // 2).item()
    h_min = (z_m - patch_size // 2).item()
    h_max = (z_m + patch_size // 2).item()
    w_min = (y_m - patch_size // 2).item()
    w_max = (y_m + patch_size // 2).item()

    d_l = max(0, -d_min)
    d_r = max(0, d_max - prompt.shape[1])
    h_l = max(0, -h_min)
    h_r = max(0, h_max - prompt.shape[2])
    w_l = max(0, -w_min)
    w_r = max(0, w_max - prompt.shape[3])

    points = torch.cat([x - d_min, y - w_min, z - h_min], dim=1).unsqueeze(1).float()
    points_torch = points.to(device)

    d_min_c = max(0, d_min)
    h_min_c = max(0, h_min)
    w_min_c = max(0, w_min)

    img_patch = image[:, :, d_min_c:d_max, h_min_c:h_max, w_min_c:w_max].clone()
    img_patch = F.pad(img_patch, (w_l, w_r, h_l, h_r, d_l, d_r))

    pred = _model_predict(
        img_patch, points_torch, img_encoder, prompt_encoder_list, mask_decoder, patch_size, device
    )
    pred = pred[:, :, d_l:patch_size - d_r, h_l:patch_size - h_r, w_l:patch_size - w_r]
    pred = F.softmax(pred, dim=1)[:, 1]

    seg_pred = torch.zeros_like(prompt).to(device)
    seg_pred[:, d_min_c:d_max, h_min_c:h_max, w_min_c:w_max] += pred

    final_pred = F.interpolate(seg_pred.unsqueeze(1), size=seg.shape[2:], mode="trilinear")
    mask = (final_pred[0, 0] > 0.5).to(torch.uint8).cpu().numpy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "pred.nii.gz"
    save_nifti(mask, reference_image, pred_path)
    return str(pred_path)
