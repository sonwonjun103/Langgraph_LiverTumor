"""
Multi-phase CT Registration
- output_path/register/attempt_{n}/ 에 저장
"""
import os
import numpy as np
import SimpleITK as sitk
from datetime import datetime
from typing import Tuple, Dict, Optional


class Registration:
    def __init__(self,
                 regis_param,
                 input_folder: str,
                 output_path: str,
                 attempt: int = 0):
        """
        Args:
            regis_param  : Registration configuration dict
            input_folder : Patient input folder (resample 폴더 포함)
            output_path  : 결과 저장 루트 (e.g. ./results/1043712)
            attempt      : 0-indexed attempt number
        """
        self.input_folder  = input_folder
        self.output_folder = os.path.join(output_path, "registration")
        os.makedirs(self.output_folder, exist_ok=True)

        self.regis_param   = regis_param
        self.attempt       = attempt
        self.current_config = {}

    # ── Registration 내부 메서드 (변경 없음) ──────────────
    def smooth_and_resample(self, image, shrink_factor, smoothing_sigma):
        if smoothing_sigma > 0:
            smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)
        else:
            smoothed_image = image
        original_spacing = image.GetSpacing()
        original_size    = image.GetSize()
        new_size    = [int(sz / float(shrink_factor) + 0.5) for sz in original_size]
        new_spacing = [
            ((original_sz - 1) * original_spc) / (new_sz - 1)
            for original_sz, original_spc, new_sz
            in zip(original_size, original_spacing, new_size)
        ]
        return sitk.Resample(
            smoothed_image, new_size, sitk.Transform(),
            sitk.sitkLinear, image.GetOrigin(),
            new_spacing, image.GetDirection(), 0.0, image.GetPixelID()
        )

    def affine_registration(self, fixed_image, moving_image, initial_transform, params):
        reg = sitk.ImageRegistrationMethod()
        reg.SetInterpolator(sitk.sitkBSpline)
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=params['histogram_bins'])
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(params['sampling_percentage'])
        reg.SetOptimizerAsGradientDescent(
            learningRate=params['learning_rate'],
            numberOfIterations=params['max_iterations'],
            convergenceMinimumValue=params['convergence_value'],
            convergenceWindowSize=params['convergence_window']
        )
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel(shrinkFactors=params['shrink_factors'])
        reg.SetSmoothingSigmasPerLevel(smoothingSigmas=params['smoothing_sigmas'])
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
        demons_filter.SetNumberOfIterations(params['iterations'])
        demons_filter.SetSmoothDisplacementField(True)
        demons_filter.SetStandardDeviations(params['sigma'])

        shrink_factors   = params['shrink_factors']
        smoothing_sigmas = params['smoothing_sigmas']

        fixed_images  = [fixed_image]
        moving_images = [moving_image]

        if shrink_factors:
            for sf, ss in reversed(list(zip(shrink_factors, smoothing_sigmas))):
                fixed_images.append(self.smooth_and_resample(fixed_images[0], sf, ss))
                moving_images.append(self.smooth_and_resample(moving_images[0], sf, ss))

        if initial_transform:
            initial_displacement_field = sitk.TransformToDisplacementField(
                initial_transform, sitk.sitkVectorFloat64,
                fixed_images[-1].GetSize(), fixed_images[-1].GetOrigin(),
                fixed_images[-1].GetSpacing(), fixed_images[-1].GetDirection()
            )
        else:
            initial_displacement_field = sitk.Image(
                fixed_images[-1].GetWidth(), fixed_images[-1].GetHeight(),
                fixed_images[-1].GetDepth(), sitk.sitkVectorFloat64
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
            portal_image, moving_image,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        affine_transform = self.affine_registration(
            portal_image, moving_image, initial_transform, config['affine']
        )
        deformable_transform = self.deformable_registration(
            fixed_image=portal_image, moving_image=moving_image,
            initial_transform=affine_transform, params=config['deformable']
        )
        registered_image = sitk.Resample(
            moving_image, portal_image, deformable_transform,
            sitk.sitkBSpline, 0.0, moving_image.GetPixelID()
        )
        registered_image.CopyInformation(portal_image)

        elapsed_time = (datetime.now() - start_time).total_seconds()
        registration_info = {
            'phase': phase_name,
            'config': config['name'],
            'elapsed_time': elapsed_time,
            'timestamp': datetime.now().isoformat()
        }
        return registered_image, registration_info

    # ── 메인 실행 ─────────────────────────────────────────
    def run(self):
        attempt_dir = os.path.join(self.output_folder, f"attempt_{self.attempt + 1}")
        os.makedirs(attempt_dir, exist_ok=True)

        # input 탐색: output_path/resample/ 우선, 없으면 input_folder/
        resample_folder = os.path.join(self.input_folder, "resample")
        if os.path.exists(resample_folder):
            arterial_path = os.path.join(resample_folder, "A.nii.gz")
            portal_path   = os.path.join(resample_folder, "P.nii.gz")
            delayed_path  = os.path.join(resample_folder, "D.nii.gz")
        else:
            arterial_path = os.path.join(self.input_folder, "A.nii.gz")
            portal_path   = os.path.join(self.input_folder, "P.nii.gz")
            delayed_path  = os.path.join(self.input_folder, "D.nii.gz")

        arterial_image = sitk.ReadImage(arterial_path)
        portal_image   = sitk.ReadImage(portal_path)
        delayed_image  = sitk.ReadImage(delayed_path)

        arterial_registered, _ = self.register_phase_to_portal(
            portal_image, arterial_image, "Arterial"
        )
        delayed_registered, _ = self.register_phase_to_portal(
            portal_image, delayed_image, "Delayed"
        )

        # output_path/register/attempt_{n}/ 에 저장
        sitk.WriteImage(portal_image,        os.path.join(attempt_dir, "P.nii.gz"))
        sitk.WriteImage(arterial_registered, os.path.join(attempt_dir, "A.nii.gz"))
        sitk.WriteImage(delayed_registered,  os.path.join(attempt_dir, "D.nii.gz"))

        print(f"  ✓ Saved registration results: {attempt_dir}")
        return attempt_dir


if __name__ == "__main__":
    param = {
        'name': 'Default',
        'affine': {
            'histogram_bins': 20, 'sampling_percentage': 0.05,
            'learning_rate': 0.1, 'max_iterations': 200,
            'convergence_value': 1e-8, 'convergence_window': 2,
            'shrink_factors': [8, 4, 2, 1], 'smoothing_sigmas': [3, 2, 1, 0]
        },
        'deformable': {
            'iterations': 100, 'sigma': 1.5,
            'shrink_factors': [8, 4, 2, 1], 'smoothing_sigmas': [3, 2, 1, 0]
        }
    }
    r = Registration(
        param,
        input_folder="/mnt/d/research/Liver/HCC_nifti_crop/1043712",
        output_path="./results/1043712",
        attempt=0
    )
    r.run()
