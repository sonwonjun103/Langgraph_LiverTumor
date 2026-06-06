import numpy as np
import torch

from torch.utils.data import Dataset


def read_sitk_volume(path):
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def normalize_volume(volume, window):
    from monai.transforms import ScaleIntensityRange

    low, high = window
    transform = ScaleIntensityRange(
        a_min=low,
        a_max=high,
        b_min=0.0,
        b_max=1.0,
        clip=True,
    )
    return transform(volume.astype(np.float32))


class CustomDataset(Dataset):
    def __init__(self,
                 A, P, D, label,
                 window=(0, 150),
                 label_threshold=0,
                 mode="val",
                 roi_size=(96, 128, 128),
                 samples_per_volume=1,
                 pos_ratio=0.8,
                 random_crop=True,
                 augment=False):
        self.A = A
        self.P = P
        self.D = D
        self.label = label
        self.window = window
        self.label_threshold = label_threshold
        self.mode = mode
        self.roi_size = tuple(roi_size)
        self.samples_per_volume = samples_per_volume if mode == "train" else 1
        self.pos_ratio = pos_ratio
        self.random_crop = random_crop
        self.augment = augment
        self.transforms = self.build_transforms()

    def build_transforms(self):
        from monai.transforms import (
            Compose,
            EnsureTyped,
            Lambdad,
            RandAdjustContrastd,
            RandAffined,
            RandCropByPosNegLabeld,
            RandFlipd,
            RandGaussianNoised,
            RandGaussianSmoothd,
            RandRotate90d,
            RandScaleIntensityd,
            RandShiftIntensityd,
            ScaleIntensityRanged,
            SpatialPadd,
        )

        low, high = self.window
        transforms = [
            ScaleIntensityRanged(
                keys=["image"],
                a_min=low,
                a_max=high,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Lambdad(keys=["label"], func=lambda x: (x > self.label_threshold).astype(np.float32)),
        ]

        if self.mode == "train":
            if self.random_crop:
                transforms.extend([
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=self.roi_size,
                        mode="constant",
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=self.roi_size,
                        pos=self.pos_ratio,
                        neg=1.0 - self.pos_ratio,
                        num_samples=1,
                        image_key="image",
                        image_threshold=0,
                    ),
                ])

            if self.augment:
                transforms.extend([
                    # Geometric
                    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
                    RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
                    RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
                    RandRotate90d(keys=["image", "label"], spatial_axes=(1, 2), prob=0.5, max_k=3),
                    RandAffined(
                        keys=["image", "label"],
                        prob=0.2,
                        rotate_range=(0.3, 0.3, 0.3),
                        scale_range=(0.2, 0.2, 0.2),
                        mode=("bilinear", "nearest"),
                        padding_mode="border",
                    ),
                    # Intensity
                    RandGaussianNoised(keys="image", prob=0.15, std=0.05),
                    RandGaussianSmoothd(
                        keys="image",
                        prob=0.15,
                        sigma_x=(0.5, 1.0),
                        sigma_y=(0.5, 1.0),
                        sigma_z=(0.5, 1.0),
                    ),
                    RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
                    RandShiftIntensityd(keys="image", offsets=0.05, prob=0.15),
                    RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.7, 1.5)),
                ])

        transforms.append(EnsureTyped(keys=["image", "label"], dtype=torch.float32))
        return Compose(transforms)

    def __len__(self):
        return len(self.A) * self.samples_per_volume

    def __getitem__(self, index):
        index = index % len(self.A)

        arterial = read_sitk_volume(self.A[index]).astype(np.float32)
        portal = read_sitk_volume(self.P[index]).astype(np.float32)
        delayed = read_sitk_volume(self.D[index]).astype(np.float32) 
        label = read_sitk_volume(self.label[index]).astype(np.float32)

        data = {
            "image": np.stack([arterial, portal, delayed], axis=0),
            "label": label[np.newaxis, ...],
        }
        data = self.transforms(data)
        if isinstance(data, list):
            data = data[0]

        return {
            "image": data["image"],
            "label": data["label"],
        }
