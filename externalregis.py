import gc
import os
from datetime import datetime

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from totalsegmentator.python_api import totalsegmentator


BASE_PATH = "/mnt/d/research/Liver/register_YU"


# ============================================================================
# REGISTRATION PARAMETERS
# ============================================================================
REGISTRATION_CONFIGS = {
    1: {
        "name": "Attempt 1",
        "affine": {
            "histogram_bins": 20,
            "sampling_percentage": 0.05,
            "learning_rate": 0.1,
            "max_iterations": 200,
            "convergence_value": 1e-8,
            "convergence_window": 2,
            "shrink_factors": [8, 4, 2, 1],
            "smoothing_sigmas": [3, 2, 1, 0],
        },
        "deformable": {
            "iterations": 50,
            "sigma": 2.0,
            "shrink_factors": [2, 1],
            "smoothing_sigmas": [1, 0],
        },
    },
}


class Resampler:
    def __init__(self):
        self.target_size = (384, 384, 96)
        self.target_resolution = (0.8, 0.8, 2.5)

    def find_common_z_range(self, nifti_files: list):
        z_mins, z_maxs = [], []
        for nii_path in nifti_files:
            img = nib.load(nii_path)
            affine = img.affine
            shape = img.shape
            z_start = affine[2, 3]
            z_end = affine[2, 3] + affine[2, 2] * (shape[2] - 1)
            z_mins.append(min(z_start, z_end))
            z_maxs.append(max(z_start, z_end))
        return max(z_mins), min(z_maxs)

    def calc_ranges(self, src_size, tgt_size, center):
        half = tgt_size // 2
        tgt_start = 0
        tgt_end = tgt_size
        src_start = center - half
        src_end = src_start + tgt_size
        if src_start < 0:
            tgt_start = -src_start
            src_start = 0
        if src_end > src_size:
            tgt_end = tgt_size - (src_end - src_size)
            src_end = src_size
        return src_start, src_end, tgt_start, tgt_end

    def crop_or_pad_volume(self, data, target_size, center_voxel):
        tx, ty, tz = target_size
        cx, cy, cz = center_voxel
        sx, sy, sz = data.shape[:3]
        output = np.full((tx, ty, tz), -1000, dtype=data.dtype)
        sx_s, sx_e, tx_s, tx_e = self.calc_ranges(sx, tx, cx)
        sy_s, sy_e, ty_s, ty_e = self.calc_ranges(sy, ty, cy)
        sz_s, sz_e, tz_s, tz_e = self.calc_ranges(sz, tz, cz)
        output[tx_s:tx_e, ty_s:ty_e, tz_s:tz_e] = data[
            sx_s:sx_e, sy_s:sy_e, sz_s:sz_e
        ]
        return output

    def resample_nifti(self, nifti_img, target_resolution):
        data = nifti_img.get_fdata()
        affine = nifti_img.affine.copy()
        header = nifti_img.header.copy()
        current_spacing = header.get_zooms()[:3]
        zoom_factors = [current_spacing[i] / target_resolution[i] for i in range(3)]
        resampled_data = zoom(data, zoom_factors, order=1)
        for i in range(3):
            affine[:3, i] = affine[:3, i] / current_spacing[i] * target_resolution[i]
        new_img = nib.Nifti1Image(resampled_data.astype(np.float32), affine)
        new_zooms = tuple(target_resolution) + header.get_zooms()[3:]
        new_img.header.set_zooms(new_zooms)
        return new_img

    def run(self, A, P, D):
        path = []
        path.append(A)
        path.append(P)
        path.append(D)

        common_z_min, common_z_max = self.find_common_z_range(path)
        common_z_center = (common_z_min + common_z_max) / 2

        resample_imgs = []
        last_output_path = None
        for nii_path in path:
            filename = os.path.basename(nii_path)
            img = nib.load(nii_path)
            resample_img = self.resample_nifti(img, self.target_resolution)
            data = resample_img.get_fdata()

            affine = resample_img.affine.copy()
            inv_affine = np.linalg.inv(affine)
            world_center = np.array([0, 0, common_z_center, 1])
            voxel_center = inv_affine @ world_center

            cx = data.shape[0] // 2
            cy = data.shape[1] // 2
            cz = int(round(voxel_center[2]))

            output_data = self.crop_or_pad_volume(
                data, self.target_size, (cx, cy, cz)
            )

            new_affine = affine.copy()
            half_x = self.target_size[0] // 2
            half_y = self.target_size[1] // 2
            half_z = self.target_size[2] // 2
            new_origin = affine @ np.array([cx - half_x, cy - half_y, cz - half_z, 1])
            new_affine[:3, 3] = new_origin[:3]

            output_img = nib.Nifti1Image(output_data.astype(np.float32), new_affine)

            resample_imgs.append(output_img)

        return resample_imgs


class Registration:
    def __init__(self, regis_param, attempt):
        self.regis_param = regis_param
        self.attempt = attempt

    def smooth_and_resample(self, image, shrink_factor, smoothing_sigma):
        if smoothing_sigma > 0:
            smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)
        else:
            smoothed_image = image
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [int(sz / float(shrink_factor) + 0.5) for sz in original_size]
        new_spacing = [
            ((original_sz - 1) * original_spc) / (new_sz - 1)
            for original_sz, original_spc, new_sz in zip(
                original_size, original_spacing, new_size
            )
        ]
        return sitk.Resample(
            smoothed_image,
            new_size,
            sitk.Transform(),
            sitk.sitkLinear,
            image.GetOrigin(),
            new_spacing,
            image.GetDirection(),
            0.0,
            image.GetPixelID(),
        )

    def affine_registration(self, fixed_image, moving_image, initial_transform, params):
        reg = sitk.ImageRegistrationMethod()
        reg.SetInterpolator(sitk.sitkBSpline)
        reg.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=params["histogram_bins"]
        )
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(params["sampling_percentage"])
        reg.SetOptimizerAsGradientDescent(
            learningRate=params["learning_rate"],
            numberOfIterations=params["max_iterations"],
            convergenceMinimumValue=params["convergence_value"],
            convergenceWindowSize=params["convergence_window"],
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel(shrinkFactors=params["shrink_factors"])
        reg.SetSmoothingSigmasPerLevel(smoothingSigmas=params["smoothing_sigmas"])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        optimized_transform = sitk.AffineTransform(3)
        reg.SetMovingInitialTransform(initial_transform)
        reg.SetInitialTransform(optimized_transform, inPlace=False)
        return reg.Execute(fixed_image, moving_image)

    def deformable_registration(self, fixed_image, moving_image, initial_transform, params):
        moving_image.SetOrigin(fixed_image.GetOrigin())
        moving_image.SetSpacing(fixed_image.GetSpacing())
        moving_image.SetDirection(fixed_image.GetDirection())

        demons_filter = sitk.FastSymmetricForcesDemonsRegistrationFilter()
        demons_filter.SetNumberOfIterations(params["iterations"])
        demons_filter.SetSmoothDisplacementField(True)
        demons_filter.SetStandardDeviations(params["sigma"])

        shrink_factors = params["shrink_factors"]
        smoothing_sigmas = params["smoothing_sigmas"]

        fixed_images = [fixed_image]
        moving_images = [moving_image]

        if shrink_factors:
            for sf, ss in reversed(list(zip(shrink_factors, smoothing_sigmas))):
                fixed_images.append(self.smooth_and_resample(fixed_images[0], sf, ss))
                moving_images.append(self.smooth_and_resample(moving_images[0], sf, ss))

        if initial_transform:
            initial_displacement_field = sitk.TransformToDisplacementField(
                initial_transform,
                sitk.sitkVectorFloat64,
                fixed_images[-1].GetSize(),
                fixed_images[-1].GetOrigin(),
                fixed_images[-1].GetSpacing(),
                fixed_images[-1].GetDirection(),
            )
        else:
            initial_displacement_field = sitk.Image(
                fixed_images[-1].GetWidth(),
                fixed_images[-1].GetHeight(),
                fixed_images[-1].GetDepth(),
                sitk.sitkVectorFloat64,
            )
            initial_displacement_field.CopyInformation(fixed_images[-1])

        initial_displacement_field = demons_filter.Execute(
            fixed_images[-1], moving_images[-1], initial_displacement_field
        )

        for f_image, m_image in reversed(list(zip(fixed_images[:-1], moving_images[:-1]))):
            initial_displacement_field = sitk.Resample(initial_displacement_field, f_image)
            initial_displacement_field = demons_filter.Execute(
                f_image, m_image, initial_displacement_field
            )

        return sitk.DisplacementFieldTransform(initial_displacement_field)

    def register_phase_to_portal(self, portal_image, moving_image, phase_name):
        config = self.regis_param
        self.current_config = config
        start_time = datetime.now()

        initial_transform = sitk.CenteredTransformInitializer(
            portal_image,
            moving_image,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        affine_transform = self.affine_registration(
            portal_image, moving_image, initial_transform, config["affine"]
        )
        deformable_transform = self.deformable_registration(
            fixed_image=portal_image,
            moving_image=moving_image,
            initial_transform=affine_transform,
            params=config["deformable"],
        )
        registered_image = sitk.Resample(
            moving_image,
            portal_image,
            deformable_transform,
            sitk.sitkBSpline,
            0.0,
            moving_image.GetPixelID(),
        )
        registered_image.CopyInformation(portal_image)

        elapsed_time = (datetime.now() - start_time).total_seconds()
        registration_info = {
            "phase": phase_name,
            "config": config["name"],
            "elapsed_time": elapsed_time,
            "timestamp": datetime.now().isoformat(),
        }
        return registered_image, registration_info

    def run(self, rA, rP, rD):
        arterial_image = sitk.ReadImage(rA)
        portal_image = sitk.ReadImage(rP)
        delayed_image = sitk.ReadImage(rD)

        arterial_registered, _ = self.register_phase_to_portal(
            portal_image, arterial_image, "Arterial"
        )

        delayed_registered, _ = self.register_phase_to_portal(
            portal_image, delayed_image, "Delayed"
        )

        return arterial_registered, portal_image, delayed_registered


def dicom_to_nifti(dicom_dir, output_path):
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in {dicom_dir}")

    series_files = [
        sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir, series_id)
        for series_id in series_ids
    ]
    dicom_files = max(series_files, key=len)

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)
    image = reader.Execute()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(image, output_path)
    return output_path


def extract_segmentation(input_path, output_path):
    totalsegmentator(
        input=input_path,
        output=output_path,
        ml=False,
        fast=True,
        verbose=False,
        roi_subset=["liver"],
    )


def compute_dice(pred, label):
    pred = pred.astype(bool)
    label = label.astype(bool)
    intersection = np.logical_and(pred, label).sum()
    denominator = pred.sum() + label.sum()
    if denominator == 0:
        return 1.0
    return 2 * intersection / denominator


def save_resampled_images(resample_imgs, resample_dir):
    os.makedirs(resample_dir, exist_ok=True)
    paths = {}
    for phase, image in zip(("A", "P", "D"), resample_imgs):
        output_path = os.path.join(resample_dir, f"{phase}.nii.gz")
        nib.save(image, output_path)
        paths[phase] = output_path
    return paths


def save_registered_images(registered_imgs, register_dir):
    os.makedirs(register_dir, exist_ok=True)
    paths = {}
    for phase, image in zip(("A", "P", "D"), registered_imgs):
        output_path = os.path.join(register_dir, f"{phase}.nii.gz")
        sitk.WriteImage(image, output_path)
        paths[phase] = output_path
    return paths


def compute_registered_liver_dice(registered_paths, register_dir):
    liverpath = os.path.join(register_dir, "liver")
    os.makedirs(liverpath, exist_ok=True)

    extract_segmentation(registered_paths["A"], os.path.join(liverpath, "A"))
    extract_segmentation(registered_paths["P"], os.path.join(liverpath, "P"))
    extract_segmentation(registered_paths["D"], os.path.join(liverpath, "D"))

    liverA = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(liverpath, "A", "liver.nii.gz"))
    )
    liverP = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(liverpath, "P", "liver.nii.gz"))
    )
    liverD = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(liverpath, "D", "liver.nii.gz"))
    )

    APdice = compute_dice(liverA, liverP)
    ADdice = compute_dice(liverA, liverD)
    PDdice = compute_dice(liverP, liverD)
    return APdice, ADdice, PDdice


def iter_case_dirs(base_path):
    for folder_name in sorted(os.listdir(base_path)):
        case_dir = os.path.join(base_path, folder_name)
        if not os.path.isdir(case_dir):
            continue

        phase_dirs = {
            phase: os.path.join(case_dir, phase)
            for phase in ("A", "P", "D")
        }
        if all(os.path.isdir(path) for path in phase_dirs.values()):
            yield folder_name, case_dir, phase_dirs
        else:
            print(f"[missing A/P/D] {case_dir}")


def process_case(folder_name, case_dir, phase_dirs):
    print(f"\nStart {folder_name}")

    dicom_nifti_dir = os.path.join(case_dir, "dicom_nifti")
    resample_dir = os.path.join(case_dir, "resample")
    register_dir = os.path.join(case_dir, "register")

    raw_paths = {}
    for phase in ("A", "P", "D"):
        raw_paths[phase] = dicom_to_nifti(
            phase_dirs[phase],
            os.path.join(dicom_nifti_dir, f"{phase}.nii.gz"),
        )

    resampler = Resampler()
    resample_imgs = resampler.run(raw_paths["A"], raw_paths["P"], raw_paths["D"])
    resampled_paths = save_resampled_images(resample_imgs, resample_dir)
    print(f"  resample saved: {resample_dir}")

    register_method = Registration(REGISTRATION_CONFIGS[1], 1)
    registered_imgs = register_method.run(
        resampled_paths["A"], resampled_paths["P"], resampled_paths["D"]
    )
    registered_paths = save_registered_images(registered_imgs, register_dir)
    print(f"  register saved: {register_dir}")

    APdice, ADdice, PDdice = compute_registered_liver_dice(
        registered_paths, register_dir
    )
    meandice = (APdice + ADdice + PDdice) / 3

    dice_txt = os.path.join(register_dir, "liver_dice.txt")
    with open(dice_txt, "w", encoding="utf-8") as f:
        f.write(f"folder: {folder_name}\n")
        f.write(f"APdice: {APdice:.6f}\n")
        f.write(f"ADdice: {ADdice:.6f}\n")
        f.write(f"PDdice: {PDdice:.6f}\n")
        f.write(f"MeanDice: {meandice:.6f}\n")

    print(
        f"  liver dice AP/AD/PD/Mean: "
        f"{APdice:.3f} / {ADdice:.3f} / {PDdice:.3f} / {meandice:.3f}"
    )
    print(f"  dice saved: {dice_txt}")

    gc.collect()


def main():
    for folder_name, case_dir, phase_dirs in iter_case_dirs(BASE_PATH):
        if folder_name == '8' or folder_name == '19' or folder_name == '9':
            try:
                process_case(folder_name, case_dir, phase_dirs)
            except Exception as exc:
                print(f"[error] {folder_name}: {exc}")


if __name__ == "__main__":
    main()
