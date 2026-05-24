import numpy as np
import os
import shutil

import nibabel as nib
from totalsegmentator.python_api import totalsegmentator


class LiverExtractor:
    """Extract liver region from CT images using TotalSegmentator"""

    PHASE_MAPPING = {
        'A': 'A_liver',
        'P': 'P_liver',
        'D': 'D_liver'
    }

    def __init__(self,
                 input_folder: str,
                 output_path: str,
                 attempt: int = 1):
        """
        Args:
            input_folder : Registration attempt folder or patient input root folder
            output_path  : 결과 저장 루트 (e.g. ./results/1043712)
            attempt      : Registration attempt number (1-indexed)
        """
        self.input_folder = input_folder
        self.attempt      = attempt

        if all(os.path.exists(os.path.join(input_folder, f"{phase}.nii.gz")) for phase in ["A", "P", "D"]):
            self.regis_folder = input_folder
            self.attempt_folder = input_folder
        else:
            self.attempt_folder = os.path.join(output_path, "registration", f"attempt_{attempt}")
            self.regis_folder = self.attempt_folder

        # TotalSegmentator masks are saved inside each registration attempt.
        self.total_folder = os.path.join(self.attempt_folder, "liver_masks")

        # Liver-only images are saved inside each registration attempt.
        self.liver_folder = os.path.join(self.attempt_folder, "liver_images")

        os.makedirs(self.total_folder, exist_ok=True)
        os.makedirs(self.liver_folder, exist_ok=True)

    def extract_segmentation(self, phase: str) -> bool:
        """TotalSegmentator로 liver 분할 (registered CT 사용)"""
        input_path  = os.path.join(self.regis_folder, f"{phase}.nii.gz")
        output_path = os.path.join(self.total_folder, phase)
        liver_mask_path = os.path.join(output_path, "liver.nii.gz")

        if not os.path.exists(input_path):
            print(f"  Error: Input file not found: {input_path}")
            return False
        if os.path.exists(liver_mask_path):
            print(f"  Reusing existing liver mask for phase {phase}: {liver_mask_path}")
            return True

        print(f"  Running TotalSegmentator for phase {phase}...")
        try:
            totalsegmentator(
                input=input_path,
                output=output_path,
                ml=False, fast=True, verbose=False,
                roi_subset=['liver']
            )
            return True
        except Exception as e:
            print(f"  Error in TotalSegmentator: {e}")
            return False

    def create_liver_masked_image(self, phase: str) -> bool:
        """Registered CT에 liver mask를 적용해 liver-only 이미지 생성"""
        if phase not in self.PHASE_MAPPING:
            print(f"  Error: Invalid phase '{phase}'")
            return False

        ct_path         = os.path.join(self.regis_folder, f"{phase}.nii.gz")
        liver_mask_path = os.path.join(self.total_folder, phase, "liver.nii.gz")
        output_filename = self.PHASE_MAPPING[phase]
        output_path     = os.path.join(self.liver_folder, f"{output_filename}.nii.gz")

        if not os.path.exists(ct_path):
            print(f"  Error: CT file not found: {ct_path}")
            return False
        if not os.path.exists(liver_mask_path):
            print(f"  Error: Liver mask not found: {liver_mask_path}")
            return False

        try:
            ct_nifti        = nib.load(ct_path)
            ct_data         = ct_nifti.get_fdata()
            liver_mask      = nib.load(liver_mask_path).get_fdata()
            liver_mask      = (liver_mask > 0).astype(np.float32)
            liver_only      = ct_data * liver_mask
            liver_nifti     = nib.Nifti1Image(
                liver_only.astype(np.float32),
                affine=ct_nifti.affine,
                header=ct_nifti.header
            )
            nib.save(liver_nifti, output_path)
            print(f"  ✓ Saved liver-masked image: {output_filename}.nii.gz")
            return True
        except Exception as e:
            print(f"  Error creating liver-masked image: {e}")
            return False

    def run(self, phase: str = None):
        if phase is None:
            return self.run_all()

        print(f"\n  Processing phase {phase}...")
        if not self.extract_segmentation(phase):
            return False
        if not self.create_liver_masked_image(phase):
            return False
        print(f"  ✓ Phase {phase} completed")
        return True

    def run_all(self) -> dict:
        results = {}
        mask_paths = {}
        print("\n" + "=" * 60)
        print("Liver Extraction Pipeline")
        print("=" * 60)
        print(f"Registration folder : {self.regis_folder}")
        print(f"Liver output folder : {self.liver_folder}")
        print("=" * 60)

        for phase in ['A', 'P', 'D']:
            results[phase] = self.run(phase)
            if results[phase]:
                mask_paths[phase] = os.path.join(self.total_folder, phase, "liver.nii.gz")

        portal_mask = mask_paths.get("P")
        if portal_mask and os.path.exists(portal_mask):
            representative_mask = os.path.join(self.attempt_folder, "liver.nii.gz")
            shutil.copy2(portal_mask, representative_mask)
            print(f"  ✓ Saved representative liver mask: {representative_mask}")

        success_count = sum(results.values())
        print("\n" + "=" * 60)
        print(f"Summary: {success_count}/3 phases completed successfully")
        print("=" * 60)
        if success_count != 3:
            return {}
        return mask_paths

    def verify_outputs(self) -> bool:
        expected_files = [f"{self.PHASE_MAPPING[p]}.nii.gz" for p in ['A', 'P', 'D']]
        all_exist = True
        for filename in expected_files:
            filepath = os.path.join(self.liver_folder, filename)
            if os.path.exists(filepath):
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} - NOT FOUND")
                all_exist = False
        return all_exist


if __name__ == '__main__':
    extractor = LiverExtractor(
        input_folder="/mnt/d/research/Liver/HCC_nifti_crop/1043712",
        output_path="./results/1043712",
        attempt=1
    )
    results = extractor.run_all()
    print("\nVerifying outputs:")
    extractor.verify_outputs()
