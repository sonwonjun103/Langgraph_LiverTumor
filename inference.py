import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from main import get_model
from train.data.dataset import normalize_volume


def read_volume(path):
    import SimpleITK as sitk

    image = sitk.ReadImage(path)
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
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key.removeprefix("module.")
        cleaned_state_dict[cleaned_key] = value

    model.load_state_dict(cleaned_state_dict)
    return checkpoint


def save_nifti(array, reference_image, output_path):
    import SimpleITK as sitk

    output_image = sitk.GetImageFromArray(array)
    output_image.CopyInformation(reference_image)
    sitk.WriteImage(output_image, output_path)


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


@torch.no_grad()
def run_inference(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = get_model(args)
    model.to(device)
    model.eval()
    load_checkpoint(model, args.checkpoint, device)

    image, reference_image = load_case(args.A, args.P, args.D, args.window)
    image = image.to(device)

    if args.use_sliding_window:
        logits = sliding_window_predict(
            model,
            image,
            args.roi_size,
            args.sw_batch_size,
            args.sw_overlap,
            args.sw_mode,
        )
    else:
        logits = model(image)
    if logits.shape[2:] != image.shape[2:]:
        logits = F.interpolate(logits, size=image.shape[2:], mode="trilinear", align_corners=False)

    probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (probability >= args.threshold).astype(np.uint8)

    os.makedirs(args.output_dir, exist_ok=True)
    mask_path = os.path.join(args.output_dir, args.output_name)
    save_nifti(mask, reference_image, mask_path)

    if args.save_probability:
        prob_path = os.path.join(args.output_dir, args.output_name.replace(".nii.gz", "_prob.nii.gz"))
        save_nifti(probability.astype(np.float32), reference_image, prob_path)

    print(f"Saved mask: {mask_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["unet", "swinunetr", "sam_adapter"],
        default="unet",
        help="Model architecture used for training.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--A", required=True, help="Arterial phase NIfTI path.")
    parser.add_argument("--P", required=True, help="Portal phase NIfTI path.")
    parser.add_argument("--D", required=True, help="Delayed phase NIfTI path.")
    parser.add_argument("--output_dir", default="./inference_outputs")
    parser.add_argument("--output_name", default="tumor_mask.nii.gz")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_probability", action="store_true")
    parser.add_argument("--window", type=float, nargs=2, default=(-200, 300))
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 128, 128])
    parser.add_argument("--use_sliding_window", action="store_true", default=True)
    parser.add_argument("--no_sliding_window", dest="use_sliding_window", action="store_false")
    parser.add_argument("--sw_batch_size", type=int, default=4)
    parser.add_argument("--sw_overlap", type=float, default=0.5)
    parser.add_argument("--sw_mode", choices=["constant", "gaussian"], default="gaussian")

    # 3D UNet arguments. These must match the training run.
    parser.add_argument("--channels", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--strides", type=int, nargs="+", default=[2, 2, 2, 2])
    parser.add_argument("--num_res_units", type=int, default=2)

    # SwinUNETR arguments. These must match the training run.
    parser.add_argument("--img_size", type=int, nargs=3, default=[96, 384, 384])
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--use_checkpoint", action="store_true")

    # 3DSAM-adapter-style arguments. These must match the training run.
    parser.add_argument("--sam_patch_size", type=int, default=16)
    parser.add_argument("--sam_tubelet_size", type=int, default=16)
    parser.add_argument("--sam_embed_dim", type=int, default=128)
    parser.add_argument("--sam_depth", type=int, default=4)
    parser.add_argument("--sam_num_heads", type=int, default=4)
    parser.add_argument("--sam_encoder_channels", type=int, default=128)
    parser.add_argument("--sam_decoder_channels", type=int, default=64)
    parser.add_argument("--sam_adapter_ratio", type=float, default=0.5)

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
