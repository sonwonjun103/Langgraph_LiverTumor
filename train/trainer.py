import json
import os
import time

import numpy as np
import torch
import tqdm

from torch.utils.data import DataLoader


class Trainer:
    def __init__(self,
                 args,
                 model,
                 A, P, D, label,
                 val_A=None, val_P=None, val_D=None, val_label=None,
                 logger=None):
        self.args = args
        self.model = model
        self.A = A
        self.P = P
        self.D = D
        self.label = label
        self.val_A = val_A
        self.val_P = val_P
        self.val_D = val_D
        self.val_label = val_label
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = None
        self.dataloader = None
        self.val_dataset = None
        self.val_dataloader = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None

    def get_dataset_class(self):
        try:
            from train.data.dataset import CustomDataset
        except ModuleNotFoundError as exc:
            if exc.name != "train":
                raise
            from data.dataset import CustomDataset

        return CustomDataset

    def get_dataset(self):
        dataset_class = self.get_dataset_class()
        self.dataset = dataset_class(self.A,
                                     self.P,
                                     self.D,
                                     self.label,
                                     window=getattr(self.args, "window", (0, 150)),
                                     mode="train",
                                     roi_size=getattr(self.args, "roi_size", (96, 128, 128)),
                                     samples_per_volume=getattr(self.args, "num_samples", 1),
                                     pos_ratio=getattr(self.args, "pos_ratio", 0.5),
                                     augment=getattr(self.args, "augment", False))
        return self.dataset
    
    def get_val_dataset(self):
        if self.val_A is None:
            return None

        dataset_class = self.get_dataset_class()
        self.val_dataset = dataset_class(self.val_A,
                                         self.val_P,
                                         self.val_D,
                                         self.val_label,
                                         window=getattr(self.args, "window", (0, 150)),
                                         mode="val",
                                         roi_size=getattr(self.args, "roi_size", (96, 128, 128)))
        return self.val_dataset

    def get_dataloader(self):
        if self.dataset is None:
            self.get_dataset()

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=getattr(self.args, "batch_size", 1),
            shuffle=True,
            num_workers=getattr(self.args, "num_workers", 0),
            pin_memory=self.device.type == "cuda",
        )
        return self.dataloader

    def get_val_dataloader(self):
        if self.val_A is None:
            return None

        if self.val_dataset is None:
            self.get_val_dataset()

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=getattr(self.args, "num_workers", 0),
            pin_memory=self.device.type == "cuda",
        )
        return self.val_dataloader

    def get_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=getattr(self.args, "lr", 1e-4),
            weight_decay=getattr(self.args, "weight_decay", 0.0),
        )
        return self.optimizer

    def get_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=getattr(self.args, "epochs", 1),
        )
        return self.scheduler

    def get_loss(self):
        from monai.losses import DiceCELoss

        self.criterion = DiceCELoss(
            sigmoid=True,
            squared_pred=True,
            lambda_dice=getattr(self.args, "dice_weight", 1.0),
            lambda_ce=getattr(self.args, "bce_weight", 1.0),
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )
        return self.criterion

    def _split_batch(self, batch):
        if isinstance(batch, dict):
            image = batch["image"]
            target = batch["label"]
        else:
            *images, target = batch
            image = torch.cat(images, dim=1) if len(images) > 1 else images[0]

        return image.float().to(self.device), target.float().to(self.device)

    @torch.no_grad()
    def sliding_window_predict(self, image):
        from monai.inferers import sliding_window_inference

        return sliding_window_inference(
            inputs=image,
            roi_size=tuple(getattr(self.args, "roi_size", (96, 128, 128))),
            sw_batch_size=getattr(self.args, "sw_batch_size", 4),
            predictor=self.model,
            overlap=getattr(self.args, "sw_overlap", 0.5),
            mode=getattr(self.args, "sw_mode", "gaussian"),
        )

    def _connected_components(self, mask):
        from scipy.ndimage import label

        structure = np.ones((3, 3, 3), dtype=np.uint8)
        labeled_mask, num_components = label(mask.astype(bool), structure=structure)
        return labeled_mask, num_components

    def _lesion_detection_counts(self, pred_mask, target_mask, thresholds):
        pred_components, num_pred = self._connected_components(pred_mask)
        target_components, num_target = self._connected_components(target_mask)

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

    def _summarize_detection_metrics(self, counts):
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

    @torch.no_grad()
    def validate(self):
        if self.val_A is None:
            return None
        if self.val_dataloader is None:
            self.get_val_dataloader()

        self.model.eval()
        total_loss = 0.0
        from monai.metrics import DiceMetric

        dice_metric = DiceMetric(include_background=True, reduction="mean")
        detection_thresholds = tuple(getattr(self.args, "detection_thresholds", [0.1, 0.2, 0.3, 0.4, 0.5]))
        detection_counts = {
            threshold: {"tp": 0, "fp": 0, "fn": 0}
            for threshold in detection_thresholds
        }

        for batch in self.val_dataloader:
            image, target = self._split_batch(batch)
            if getattr(self.args, "use_sliding_window", True):
                output = self.sliding_window_predict(image)
            else:
                output = self.model(image)
            loss = self.criterion(output, target)

            pred_mask = (torch.sigmoid(output) > 0.5).float()
            dice_metric(y_pred=pred_mask, y=target)

            pred_np = pred_mask[:, 0].cpu().numpy().astype(bool)
            target_np = (target[:, 0].cpu().numpy() > 0.5)
            for sample_index in range(pred_np.shape[0]):
                sample_counts = self._lesion_detection_counts(
                    pred_np[sample_index],
                    target_np[sample_index],
                    detection_thresholds,
                )
                for threshold in detection_thresholds:
                    detection_counts[threshold]["tp"] += sample_counts[threshold]["tp"]
                    detection_counts[threshold]["fp"] += sample_counts[threshold]["fp"]
                    detection_counts[threshold]["fn"] += sample_counts[threshold]["fn"]

            total_loss += loss.item()

        num_batches = max(len(self.val_dataloader), 1)
        mean_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        return {
            "loss": total_loss / num_batches,
            "dice": mean_dice,
            "detection": self._summarize_detection_metrics(detection_counts),
        }

    def _save_training_plot(self, history):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        output_dir = getattr(self.args, "output_dir", "checkpoints")
        os.makedirs(output_dir, exist_ok=True)

        epochs = np.arange(1, len(history) + 1)
        train_losses = [entry["train_loss"] for entry in history]
        val_losses = [entry["val"]["loss"] if entry["val"] else np.nan for entry in history]
        val_dices = [entry["val"]["dice"] if entry["val"] else np.nan for entry in history]

        has_val = any(entry["val"] is not None for entry in history)
        detection_thresholds = []
        for entry in history:
            if entry["val"] is not None:
                detection_thresholds = sorted(entry["val"]["detection"].keys())
                break
        has_detection = bool(detection_thresholds)

        n_panels = 1 + (1 if has_val else 0) + (1 if has_detection else 0)
        fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4 * n_panels))
        if n_panels == 1:
            axes = [axes]

        axes[0].plot(epochs, train_losses, label="train_loss", marker="o", markersize=3)
        if has_val:
            axes[0].plot(epochs, val_losses, label="val_loss", marker="s", markersize=3)
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        panel_idx = 1
        if has_val:
            axes[panel_idx].plot(epochs, val_dices, label="val_dice", marker="o", color="green", markersize=3)
            axes[panel_idx].set_xlabel("epoch")
            axes[panel_idx].set_ylabel("dice")
            axes[panel_idx].set_title("Val Dice")
            axes[panel_idx].set_ylim(0.0, 1.0)
            axes[panel_idx].legend()
            axes[panel_idx].grid(True, alpha=0.3)
            panel_idx += 1

        if has_detection:
            for threshold in detection_thresholds:
                f1_series = [
                    entry["val"]["detection"][threshold]["f1"] if entry["val"] else np.nan
                    for entry in history
                ]
                axes[panel_idx].plot(epochs, f1_series, label=f"f1@{threshold:.1f}", marker="o", markersize=3)
            axes[panel_idx].set_xlabel("epoch")
            axes[panel_idx].set_ylabel("f1")
            axes[panel_idx].set_title("Detection F1")
            axes[panel_idx].set_ylim(0.0, 1.0)
            axes[panel_idx].legend()
            axes[panel_idx].grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(output_dir, "training_curves.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)

        history_path = os.path.join(output_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return path

    def save_checkpoint(self, epoch, train_loss, val_metrics=None):
        output_dir = getattr(self.args, "output_dir", "checkpoints")
        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(output_dir, "best_model.pt")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, path)
        return path

    def train(self):
        if self.dataloader is None:
            self.get_dataloader()
        if self.val_A is not None and self.val_dataloader is None:
            self.get_val_dataloader()
        if self.optimizer is None:
            self.get_optimizer()
        if self.scheduler is None:
            self.get_scheduler()
        if self.criterion is None:
            self.get_loss()

        epochs = getattr(self.args, "epochs", 1)
        self.model.to(self.device)

        history = []
        best_score = -1.0
        for epoch in tqdm.tqdm(range(epochs)):
            start_time = time.time()
            running_loss = 0.0
            self.model.train()

            for batch in self.dataloader:
                image, target = self._split_batch(batch)

                self.optimizer.zero_grad(set_to_none=True)
                output = self.model(image)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / max(len(self.dataloader), 1)
            should_validate = (
                self.val_A is not None
                and (epoch + 1) % getattr(self.args, "val_interval", 1) == 0
            )
            val_metrics = self.validate() if should_validate else None
            history.append({"train_loss": epoch_loss, "val": val_metrics})
            elapsed = time.time() - start_time

            message = f"Epoch [{epoch + 1}/{epochs}] loss: {epoch_loss:.4f}"
            if val_metrics is not None:
                message += f" val_loss: {val_metrics['loss']:.4f} val_dice: {val_metrics['dice']:.4f}"
                for threshold, metrics in val_metrics["detection"].items():
                    message += (
                        f" det@{threshold:.1f}"
                        f" f1: {metrics['f1']:.4f}"
                        f" sens: {metrics['sensitivity']:.4f}"
                    )

                if val_metrics["dice"] > best_score:
                    best_score = val_metrics["dice"]
                    path = self.save_checkpoint(epoch + 1, epoch_loss, val_metrics)
                    message += f" saved: {path}"

            message += f" time: {elapsed:.1f}s"
            if self.logger is not None:
                self.logger.info(message)
            else:
                print(message)

            self._save_training_plot(history)
            self.scheduler.step()

        return history
