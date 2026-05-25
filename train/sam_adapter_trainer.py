import importlib
import json
import os
import shutil
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader


def _load_adapter_modules():
    try:
        image_encoder_module = importlib.import_module("train.models.3DSAMAdapter.image_encoder")
        mask_decoder_module = importlib.import_module("train.models.3DSAMAdapter.mask_decoder")
        prompt_encoder_module = importlib.import_module("train.models.3DSAMAdapter.prompt_encoder")
    except ModuleNotFoundError as exc:
        if exc.name == "segment_anything":
            raise ModuleNotFoundError(
                "3DSAMAdapter needs the segment_anything package. Install the SAM package "
                "and make sure ckpt/sam_vit_b_01ec64.pth is available."
            ) from exc
        raise

    return (
        image_encoder_module.ImageEncoderViT_3d_v2,
        mask_decoder_module.VIT_MLAHead_h,
        prompt_encoder_module.PromptEncoder,
        prompt_encoder_module.TwoWayTransformer,
    )


def _load_sam_image_encoder_state(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SAM checkpoint was not found: {checkpoint_path}. "
            "Pass --sam_checkpoint /path/to/sam_vit_b_01ec64.pth."
        )

    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "segment_anything is required to initialize 3DSAMAdapter from SAM ViT-B."
        ) from exc

    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    mask_generator = SamAutomaticMaskGenerator(sam)
    state_dict = mask_generator.predictor.model.image_encoder.state_dict()
    del mask_generator
    del sam
    return state_dict


def _load_pretrained_state(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"3DSAMAdapter pretrained checkpoint was not found: {path}. "
            "Pass --sam_pretrained_ckpt /path/to/last.pth.tar."
        )
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _set_encoder_trainable_layers(img_encoder):
    for parameter in img_encoder.parameters():
        parameter.requires_grad = False

    img_encoder.depth_embed.requires_grad = True
    for parameter in img_encoder.slice_embed.parameters():
        parameter.requires_grad = True

    for block in img_encoder.blocks:
        for parameter in block.norm1.parameters():
            parameter.requires_grad = True
        for parameter in block.adapter.parameters():
            parameter.requires_grad = True
        for parameter in block.norm2.parameters():
            parameter.requires_grad = True
        block.attn.rel_pos_d = nn.Parameter(
            0.5 * (block.attn.rel_pos_h + block.attn.rel_pos_w),
            requires_grad=True,
        )

    for neck in img_encoder.neck_3d:
        for parameter in neck.parameters():
            parameter.requires_grad = True


def _build_prompt_encoders(PromptEncoder, TwoWayTransformer, pretrained_state, device):
    prompt_encoder_list = []
    prompt_parameters = []

    for index in range(4):
        prompt_encoder = PromptEncoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            )
        )
        prompt_encoder.load_state_dict(pretrained_state["feature_dict"][index], strict=True)
        prompt_encoder.to(device)
        prompt_encoder_list.append(prompt_encoder)
        prompt_parameters.extend(
            parameter for parameter in prompt_encoder.parameters() if parameter.requires_grad
        )

    return prompt_encoder_list, prompt_parameters


def _sample_prompt_points(seg, num_pos, num_neg, device):
    if seg.ndim == 5:
        seg = seg[:, 0]

    positive = torch.where(seg == 1)
    points_torch = None
    if len(positive[0]) > 0:
        sample = np.random.choice(np.arange(len(positive[0])), num_pos, replace=True)
        x = positive[1][sample].unsqueeze(1)
        y = positive[3][sample].unsqueeze(1)
        z = positive[2][sample].unsqueeze(1)
        points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
        points_torch = points.to(device).transpose(0, 1)

    negative = torch.where(seg < 10)
    if len(negative[0]) == 0:
        negative = torch.where(torch.ones_like(seg, dtype=torch.bool))

    sample = np.random.choice(np.arange(len(negative[0])), num_neg, replace=True)
    x = negative[1][sample].unsqueeze(1)
    y = negative[3][sample].unsqueeze(1)
    z = negative[2][sample].unsqueeze(1)
    points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
    negative_points = points.to(device).transpose(0, 1)

    if points_torch is None:
        return negative_points
    return torch.cat([points_torch, negative_points], dim=1)


def _connected_components(mask):
    from scipy.ndimage import label

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_mask, num_components = label(mask.astype(bool), structure=structure)
    return labeled_mask, num_components


def _lesion_detection_counts(pred_mask, target_mask, thresholds):
    pred_components, num_pred = _connected_components(pred_mask)
    target_components, num_target = _connected_components(target_mask)

    counts = {
        threshold: {"tp": 0, "fp": num_pred, "fn": num_target}
        for threshold in thresholds
    }
    if num_pred == 0 or num_target == 0:
        return counts

    pairs = []
    for target_id in range(1, num_target + 1):
        target_component = target_components == target_id
        target_volume = target_component.sum()

        overlapping_pred_ids = np.unique(pred_components[target_component])
        overlapping_pred_ids = overlapping_pred_ids[overlapping_pred_ids > 0]

        for pred_id in overlapping_pred_ids:
            pred_component = pred_components == pred_id
            intersection = np.logical_and(target_component, pred_component).sum()
            union = target_volume + pred_component.sum() - intersection
            overlap = intersection / union if union > 0 else 0.0
            pairs.append((overlap, target_id, pred_id))

    pairs.sort(reverse=True)

    for threshold in thresholds:
        matched_targets = set()
        matched_preds = set()

        for overlap, target_id, pred_id in pairs:
            if overlap < threshold:
                break
            if target_id in matched_targets or pred_id in matched_preds:
                continue
            matched_targets.add(target_id)
            matched_preds.add(pred_id)

        tp = len(matched_targets)
        fp = num_pred - tp
        fn = num_target - tp
        counts[threshold] = {"tp": tp, "fp": fp, "fn": fn}

    return counts


def _summarize_detection_metrics(counts):
    metrics = {}
    for threshold, values in counts.items():
        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        sensitivity = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity > 0 else 0.0

        metrics[threshold] = {
            "precision": precision,
            "sensitivity": sensitivity,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return metrics


def _crop_prompt_patch(img, seg, patch_size, device):
    """Crop a cubic patch centered at a random GT foreground voxel.

    Returns ``(img_patch, seg_patch)`` both shaped ``(1, C, P, P, P)`` where
    ``P = patch_size``, zero-padded when the chosen center is near the volume
    boundary. Falls back to the volume center when ``seg`` has no foreground.
    """
    if seg.ndim == 4:
        seg_view = seg.unsqueeze(1)
    else:
        seg_view = seg

    D, H, W = seg_view.shape[2], seg_view.shape[3], seg_view.shape[4]
    fg = torch.where(seg_view[0, 0] == 1)
    if fg[0].numel() == 0:
        cd, ch, cw = D // 2, H // 2, W // 2
    else:
        idx = int(np.random.choice(np.arange(fg[0].numel()), 1)[0])
        cd, ch, cw = int(fg[0][idx]), int(fg[1][idx]), int(fg[2][idx])

    half = patch_size // 2
    d_min, d_max = cd - half, cd + half
    h_min, h_max = ch - half, ch + half
    w_min, w_max = cw - half, cw + half

    d_l = max(0, -d_min)
    d_r = max(0, d_max - D)
    h_l = max(0, -h_min)
    h_r = max(0, h_max - H)
    w_l = max(0, -w_min)
    w_r = max(0, w_max - W)

    d_min_c = max(0, d_min)
    h_min_c = max(0, h_min)
    w_min_c = max(0, w_min)

    img_patch = img[:, :, d_min_c:d_max, h_min_c:h_max, w_min_c:w_max].clone()
    seg_patch = seg_view[:, :, d_min_c:d_max, h_min_c:h_max, w_min_c:w_max].clone()

    img_patch = F.pad(img_patch, (w_l, w_r, h_l, h_r, d_l, d_r))
    seg_patch = F.pad(seg_patch, (w_l, w_r, h_l, h_r, d_l, d_r))

    return img_patch.to(device), seg_patch.to(device)


def _forward_3dsam(img_encoder, prompt_encoder_list, mask_decoder, img, seg, patch_size, device,
                   num_pos=10, num_neg=20):
    out = F.interpolate(
        img.float(),
        scale_factor=512 / patch_size,
        mode="trilinear",
        align_corners=False,
    )
    input_batch = out.to(device)
    input_batch = input_batch[0].transpose(0, 1)

    batch_features, feature_list = img_encoder(input_batch)
    feature_list.append(batch_features)

    points_torch = _sample_prompt_points(seg, num_pos=num_pos, num_neg=num_neg, device=device)
    new_feature = []
    for index, (feature, prompt_encoder) in enumerate(zip(feature_list, prompt_encoder_list)):
        if index == 3:
            new_feature.append(
                prompt_encoder(feature, points_torch.clone(), [patch_size, patch_size, patch_size])
            )
        else:
            new_feature.append(feature)

    img_resize = F.interpolate(
        img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device),
        scale_factor=64 / patch_size,
        mode="trilinear",
        align_corners=False,
    )
    new_feature.append(img_resize)

    masks = mask_decoder(new_feature, 2, patch_size // 64)
    return masks.permute(0, 1, 4, 2, 3)


def _save_training_plot(history, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    os.makedirs(output_dir, exist_ok=True)
    epochs = [entry["epoch"] for entry in history]
    train_losses = [entry["train_loss"] for entry in history]
    val_losses = [
        entry["val_loss"] if entry["val_loss"] is not None else float("nan")
        for entry in history
    ]
    has_val = any(entry["val_loss"] is not None for entry in history)

    detection_thresholds = []
    for entry in history:
        if entry.get("detection"):
            detection_thresholds = sorted(entry["detection"].keys())
            break
    has_detection = bool(detection_thresholds)

    n_panels = 1 + (1 if has_detection else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    axes[0].plot(epochs, train_losses, label="train_loss", marker="o", markersize=3)
    if has_val:
        axes[0].plot(epochs, val_losses, label="val_loss", marker="s", markersize=3)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("3D SAM Adapter Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if has_detection:
        for threshold in detection_thresholds:
            f1_series = [
                entry["detection"][threshold]["f1"] if entry.get("detection") else float("nan")
                for entry in history
            ]
            axes[1].plot(epochs, f1_series, label=f"f1@{threshold:.1f}", marker="o", markersize=3)
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("f1")
        axes[1].set_title("Detection F1 (patch-level)")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return path


def _save_checkpoint(state, is_best, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_path = os.path.join(checkpoint_dir, "last.pth.tar")
    torch.save(state, last_path)

    if is_best:
        best_tar_path = os.path.join(checkpoint_dir, "best.pth.tar")
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
        shutil.copyfile(last_path, best_tar_path)
        shutil.copyfile(last_path, best_model_path)

    return last_path 


def _make_loader(args, A, P, D, label, mode):
    from train.data.dataset import CustomDataset

    dataset = CustomDataset(
        A,
        P,
        D,
        label,
        window=getattr(args, "window", (-200, 300)),
        mode=mode,
        roi_size=getattr(args, "roi_size", (128, 128, 128)),
        samples_per_volume=getattr(args, "num_samples", 1) if mode == "train" else 1,
        pos_ratio=getattr(args, "pos_ratio", 0.5),
        augment=getattr(args, "augment", False) if mode == "train" else False,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=mode == "train",
        num_workers=getattr(args, "num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )


def _validate_roi_size(args):
    roi_size = tuple(getattr(args, "roi_size", (128, 128, 128)))
    if len(set(roi_size)) != 1:
        raise ValueError(
            "3DSAMAdapter reference training expects a cubic crop. "
            "Use --roi_size 128 128 128."
        )
    if roi_size[0] % 64 != 0:
        raise ValueError("3DSAMAdapter roi_size must be divisible by 64.")
    return roi_size[0]


def train_sam_adapter(args, train_paths, val_paths=None, logger=None):
    from monai.losses import DiceCELoss, DiceLoss

    ImageEncoderViT_3d, VIT_MLAHead_h, PromptEncoder, TwoWayTransformer = _load_adapter_modules()

    patch_size = _validate_roi_size(args)
    requested_device = getattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        if logger is not None:
            logger.info("CUDA was requested but is not available; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    output_dir = getattr(args, "output_dir", "./checkpoints/sam_adapter")

    if logger is not None:
        logger.info("Building 3DSAMAdapter from SAM ViT-B and pretrained 3D adapter weights.")

    sam_state = _load_sam_image_encoder_state(getattr(args, "sam_checkpoint", "./models/3DSAMAdapter/ckpt/sam_vit_b_01ec64.pth"))
    pretrained_state = _load_pretrained_state(
        getattr(args, "sam_pretrained_ckpt", "./snapshot/lits/last.pth.tar")
    )

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
    img_encoder.load_state_dict(sam_state, strict=False)
    img_encoder.load_state_dict(pretrained_state["encoder_dict"], strict=False)
    img_encoder.to(device)
    _set_encoder_trainable_layers(img_encoder)

    prompt_encoder_list, prompt_parameters = _build_prompt_encoders(
        PromptEncoder,
        TwoWayTransformer,
        pretrained_state,
        device,
    )

    mask_decoder = VIT_MLAHead_h(img_size=96, num_classes=2)
    mask_decoder.load_state_dict(pretrained_state["decoder_dict"], strict=True)
    mask_decoder.to(device)

    encoder_opt = AdamW(
        [parameter for parameter in img_encoder.parameters() if parameter.requires_grad],
        lr=getattr(args, "lr", 4e-4),
        weight_decay=0,
    )
    feature_opt = AdamW(prompt_parameters, lr=getattr(args, "lr", 4e-4), weight_decay=0)
    decoder_opt = AdamW(
        [parameter for parameter in mask_decoder.parameters() if parameter.requires_grad],
        lr=getattr(args, "lr", 4e-4),
        weight_decay=0,
    )

    max_epochs = getattr(args, "epochs", 100)
    encoder_scheduler = torch.optim.lr_scheduler.LinearLR(
        encoder_opt, start_factor=1.0, end_factor=0.01, total_iters=max_epochs
    )
    feature_scheduler = torch.optim.lr_scheduler.LinearLR(
        feature_opt, start_factor=1.0, end_factor=0.01, total_iters=max_epochs
    )
    decoder_scheduler = torch.optim.lr_scheduler.LinearLR(
        decoder_opt, start_factor=1.0, end_factor=0.01, total_iters=max_epochs
    )

    train_loader = _make_loader(args, *train_paths, mode="train")
    val_loader = _make_loader(args, *val_paths, mode="val") if val_paths is not None else None

    loss_cal = DiceCELoss(
        include_background=False,
        softmax=True,
        to_onehot_y=True,
        lambda_dice=getattr(args, "sam_dice_weight", 0.5),
        lambda_ce=getattr(args, "sam_ce_weight", 0.5),
    )
    val_loss_cal = DiceLoss(
        include_background=False,
        softmax=True,
        to_onehot_y=True,
        reduction="none",
    )

    best_loss = np.inf
    best_epoch = 0
    history = []
    eval_interval = getattr(args, "sam_eval_interval", getattr(args, "val_interval", 1))

    for epoch in range(max_epochs):
        img_encoder.train()
        mask_decoder.train()
        for module in prompt_encoder_list:
            module.train()

        train_losses = []
        for index, batch in enumerate(train_loader):
            img = batch["image"].to(device)
            seg = batch["label"].to(device).long()

            masks = _forward_3dsam(
                img_encoder,
                prompt_encoder_list,
                mask_decoder,
                img,
                seg,
                patch_size,
                device,
                num_pos=getattr(args, "sam_num_pos_points", 10),
                num_neg=getattr(args, "sam_num_neg_points", 20),
            )
            loss = loss_cal(masks, seg)

            encoder_opt.zero_grad(set_to_none=True)
            feature_opt.zero_grad(set_to_none=True)
            decoder_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(img_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(mask_decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(prompt_encoder_list[-1].parameters(), 1.0)
            encoder_opt.step()
            feature_opt.step()
            decoder_opt.step()

            train_losses.append(float(loss.detach().cpu()))
            if logger is not None:
                logger.info(
                    f"epoch: {epoch + 1}/{max_epochs}, iter: {index + 1}/{len(train_loader)} "
                    f"loss: {train_losses[-1]:.6f}"
                )

        encoder_scheduler.step()
        feature_scheduler.step()
        decoder_scheduler.step()

        mean_train_loss = float(np.mean(train_losses)) if train_losses else np.inf
        val_loss = None
        detection_metrics = None

        if val_loader is not None and (epoch + 1) % eval_interval == 0:
            img_encoder.eval()
            mask_decoder.eval()
            for module in prompt_encoder_list:
                module.eval()

            detection_thresholds = tuple(
                getattr(args, "detection_thresholds", [0.1, 0.2, 0.3, 0.4, 0.5])
            )
            detection_counts = {
                threshold: {"tp": 0, "fp": 0, "fn": 0}
                for threshold in detection_thresholds
            }

            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    img = batch["image"].to(device)
                    seg = batch["label"].to(device).long()
                    img_patch, seg_patch = _crop_prompt_patch(img, seg, patch_size, device)
                    masks = _forward_3dsam(
                        img_encoder,
                        prompt_encoder_list,
                        mask_decoder,
                        img_patch,
                        seg_patch,
                        patch_size,
                        device,
                        num_pos=getattr(args, "sam_num_pos_points", 10),
                        num_neg=getattr(args, "sam_val_num_neg_points", 10),
                    )
                    loss = val_loss_cal(masks, seg_patch)
                    val_losses.append(float(loss.detach().cpu().mean()))

                    pred_mask = (torch.softmax(masks, dim=1)[:, 1] > 0.5).cpu().numpy().astype(bool)
                    target_np = (seg_patch[:, 0].cpu().numpy() > 0.5)
                    for sample_index in range(pred_mask.shape[0]):
                        sample_counts = _lesion_detection_counts(
                            pred_mask[sample_index],
                            target_np[sample_index],
                            detection_thresholds,
                        )
                        for threshold in detection_thresholds:
                            detection_counts[threshold]["tp"] += sample_counts[threshold]["tp"]
                            detection_counts[threshold]["fp"] += sample_counts[threshold]["fp"]
                            detection_counts[threshold]["fn"] += sample_counts[threshold]["fn"]

            val_loss = float(np.mean(val_losses)) if val_losses else np.inf
            detection_metrics = _summarize_detection_metrics(detection_counts)
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                best_epoch = epoch + 1

            _save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "best_val_loss": best_loss,
                    "encoder_dict": img_encoder.state_dict(),
                    "decoder_dict": mask_decoder.state_dict(),
                    "feature_dict": [module.state_dict() for module in prompt_encoder_list],
                    "encoder_opt": encoder_opt.state_dict(),
                    "feature_opt": feature_opt.state_dict(),
                    "decoder_opt": decoder_opt.state_dict(),
                    "args": vars(args),
                },
                is_best=is_best,
                checkpoint_dir=output_dir,
            )

        history.append({
            "epoch": epoch + 1,
            "train_loss": mean_train_loss,
            "val_loss": val_loss,
            "detection": detection_metrics,
        })
        _save_training_plot(history, output_dir)
        if logger is not None:
            message = f"3DSAMAdapter epoch {epoch + 1}/{max_epochs} train_loss: {mean_train_loss:.6f}"
            if val_loss is not None:
                message += f" val_loss: {val_loss:.6f} best: {best_loss:.6f} at epoch {best_epoch}"
            if detection_metrics is not None:
                for threshold, metrics in detection_metrics.items():
                    message += (
                        f" det@{threshold:.1f}"
                        f"_f1: {metrics['f1']:.4f}"
                        f"_sens: {metrics['sensitivity']:.4f}"
                    )
            logger.info(message)

    return {
        "history": history,
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "output_dir": output_dir,
    }
