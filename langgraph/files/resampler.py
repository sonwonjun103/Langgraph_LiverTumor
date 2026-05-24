'''
input_folder에 A.nii.gz, P.nii.gz, D.nii.gz가 있다고 가정
Resampled 데이터는 output_path/resample/ 폴더에 저장
'''
import os
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


class Resampler:
    def __init__(self,
                 input_folder: str,
                 output_path: str):
        """
        Args:
            input_folder : NIfTI 파일이 있는 폴더 (A/P/D .nii.gz)
            output_path  : 결과 저장 루트 (e.g. ./results/1043712)
        """
        self.input_folder  = input_folder
        self.phase_names   = ['A', 'P', 'D']
        self.target_size   = (384, 384, 96)
        self.target_resolution = (0.8, 0.8, 2.5)

        self.output_folder = os.path.join(output_path, "resample")
        os.makedirs(self.output_folder, exist_ok=True)

    def get_nifti_files(self, folder: str):
        files = (glob.glob(os.path.join(folder, '*.nii.gz')) +
                 glob.glob(os.path.join(folder, '*.nii')))
        return sorted(files)

    def find_common_z_range(self, nifti_files: list):
        z_mins, z_maxs = [], []
        for nii_path in nifti_files:
            img    = nib.load(nii_path)
            affine = img.affine
            shape  = img.shape
            z_start = affine[2, 3]
            z_end   = affine[2, 3] + affine[2, 2] * (shape[2] - 1)
            z_mins.append(min(z_start, z_end))
            z_maxs.append(max(z_start, z_end))
        return max(z_mins), min(z_maxs)

    def calc_ranges(self, src_size, tgt_size, center):
        half      = tgt_size // 2
        tgt_start = 0
        tgt_end   = tgt_size
        src_start = center - half
        src_end   = src_start + tgt_size
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
        output[tx_s:tx_e, ty_s:ty_e, tz_s:tz_e] = data[sx_s:sx_e, sy_s:sy_e, sz_s:sz_e]
        return output

    def resample_nifti(self, nifti_img, target_resolution):
        data    = nifti_img.get_fdata()
        affine  = nifti_img.affine.copy()
        header  = nifti_img.header.copy()
        current_spacing = header.get_zooms()[:3]
        zoom_factors = [
            current_spacing[i] / target_resolution[i] for i in range(3)
        ]
        resampled_data = zoom(data, zoom_factors, order=1)
        for i in range(3):
            affine[:3, i] = affine[:3, i] / current_spacing[i] * target_resolution[i]
        new_img = nib.Nifti1Image(resampled_data.astype(np.float32), affine)
        new_zooms = tuple(target_resolution) + header.get_zooms()[3:]
        new_img.header.set_zooms(new_zooms)
        return new_img

    def run(self):
        nifti_files = self.get_nifti_files(self.input_folder)
        if not nifti_files:
            print(f"  No NIfTI files found in {self.input_folder}")
            return

        print(f"  Processing folder : {self.input_folder}")
        print(f"  Output folder     : {self.output_folder}")
        print(f"  Found {len(nifti_files)} NIfTI file(s)")

        common_z_min, common_z_max = self.find_common_z_range(nifti_files)
        common_z_center = (common_z_min + common_z_max) / 2

        last_output_path = None
        for nii_path in nifti_files:
            filename     = os.path.basename(nii_path)
            img          = nib.load(nii_path)
            resample_img = self.resample_nifti(img, self.target_resolution)
            data         = resample_img.get_fdata()

            affine       = resample_img.affine.copy()
            inv_affine   = np.linalg.inv(affine)
            world_center = np.array([0, 0, common_z_center, 1])
            voxel_center = inv_affine @ world_center

            cx = data.shape[0] // 2
            cy = data.shape[1] // 2
            cz = int(round(voxel_center[2]))

            output_data = self.crop_or_pad_volume(data, self.target_size, (cx, cy, cz))

            new_affine  = affine.copy()
            half_x = self.target_size[0] // 2
            half_y = self.target_size[1] // 2
            half_z = self.target_size[2] // 2
            new_origin = affine @ np.array([cx - half_x, cy - half_y, cz - half_z, 1])
            new_affine[:3, 3] = new_origin[:3]

            output_img  = nib.Nifti1Image(output_data.astype(np.float32), new_affine)
            output_path = os.path.join(self.output_folder, filename)
            nib.save(output_img, output_path)
            print(f"  ✓ Saved: {output_path}")
            last_output_path = output_path

        return last_output_path


if __name__ == '__main__':
    r = Resampler(
        input_folder="/home/sonwonjun/research/MGH/test/2280937",
        output_path="./results/2280937"
    )
    r.run()