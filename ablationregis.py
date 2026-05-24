import os, gc
import SimpleITK as sitk
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from totalsegmentator.python_api import totalsegmentator

# ============================================================================
# REGISTRATION PARAMETERS
# ============================================================================
REGISTRATION_CONFIGS = {
    1: {  # Attempt 1
        'name': 'Attempt 1',
        'affine': {
            'histogram_bins': 20,
            'sampling_percentage': 0.05,
            'learning_rate': 0.1,
            'max_iterations': 200,
            'convergence_value': 1e-8,
            'convergence_window': 2,
            'shrink_factors': [8, 4, 2, 1],
            'smoothing_sigmas': [3, 2, 1, 0]
        },
        'deformable': {
            'iterations': 50,
            'sigma': 2.0,
            'shrink_factors': [2, 1],
            'smoothing_sigmas': [1, 0]
        }
    },
    2: {  # Attempt 2
        'name': 'Attempt 2',
        'affine': {
            'histogram_bins': 30,
            'sampling_percentage': 0.1,
            'learning_rate': 0.05,
            'max_iterations': 400,
            'convergence_value': 1e-9,
            'convergence_window': 5,
            'shrink_factors': [8, 4, 2, 1],
            'smoothing_sigmas': [3, 2, 1, 0]
        },
        'deformable': {
            'iterations': 100,
            'sigma': 1.5,
            'shrink_factors': [4, 2, 1],
            'smoothing_sigmas': [2, 1, 0]
        }
    },
    3: {  # Attempt 3
        'name': 'Attempt 3',
        'affine': {
            'histogram_bins': 40,
            'sampling_percentage': 0.15,
            'learning_rate': 0.02,
            'max_iterations': 600,
            'convergence_value': 1e-10,
            'convergence_window': 10,
            'shrink_factors': [16, 8, 4, 2, 1],
            'smoothing_sigmas': [4, 3, 2, 1, 0]
        },
        'deformable': {
            'iterations': 150,
            'sigma': 1.0,  # =============================
            'shrink_factors': [8, 4, 2, 1],
            'smoothing_sigmas': [3, 2, 1, 0]
        }
    },
    4: {  # Attempt 4
        'name': 'Attempt 4',
        'affine': {
            'histogram_bins': 50,
            'sampling_percentage': 0.2,
            'learning_rate': 0.01,
            'max_iterations': 350,
            'convergence_value': 1e-11,
            'convergence_window': 15,
            'shrink_factors': [16, 8, 4, 2, 1],
            'smoothing_sigmas': [4, 3, 2, 1, 0]
        },
        'deformable': {
            'iterations': 200,
            'sigma': 0.5,
            'shrink_factors': [8, 4, 2, 1],
            'smoothing_sigmas': [3, 2, 1, 0]
        }
    },

    5: {  # Attempt 5
        'name': 'Attempt 5',
        'affine': {
            'histogram_bins': 60,
            'sampling_percentage': 0.3,
            'learning_rate': 0.005,
            'max_iterations': 400,
            'convergence_value': 1e-12,
            'convergence_window': 20,
            'shrink_factors': [16, 8, 4, 2, 1],
            'smoothing_sigmas': [4, 3, 2, 1, 0]
        },
        'deformable': {
            'iterations': 300,
            'sigma': 0.3,
            'shrink_factors': [8, 4, 2, 1],
            'smoothing_sigmas': [3, 2, 1, 0]
        }
    },
}

class Resampler:
    def __init__(self):
        self.target_size = (384, 384, 96)
        self.target_resolution = (0.8, 0.8, 2.5)

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
            #output_data = output_img.get_fdata()

            resample_imgs.append(output_img)
        
        return resample_imgs
    
"""
registration 된 후의 A P D
"""
import SimpleITK as sitk
from datetime import datetime

class Registration:
    def __init__(self,
                 regis_param,
                 attempt):
        self.regis_param = regis_param
        self.attempt = attempt

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
    
def extract_segmentation(input_path, output_path):
    totalsegmentator(
        input=input_path,
        output=output_path,
        ml=False, fast=True, verbose=False,
        roi_subset=['liver']
    )

def compute_dice(pred, label):
    """Compute Dice score between binary prediction and label."""
    pred = pred.astype(bool)
    label = label.astype(bool)
    intersection = np.logical_and(pred, label).sum()
    denominator = pred.sum() + label.sum()
    if denominator == 0:
        return 1.0  # Both empty → perfect match
    return 2 * intersection / denominator

def main():
    data1 = pd.read_excel("./files/data1metrics.xlsx")
    data2 = pd.read_excel("./files/data2metrics.xlsx")

    # data1: content 컬럼이 비어있고, tumor size >= 2000
    data1_filtered = data1[
        data1['content'].isna() &
        (data1['tumor_size'] >= 2000)
    ]

    # data2: tumor size >= 2000
    data2_filtered = data2[
        data2['tumor_size'] >= 2000
    ]

    print(f"data1: {len(data1)} → {len(data1_filtered)}행")
    print(f"data2: {len(data2)} → {len(data2_filtered)}행")

    data1_subject = data1_filtered['subject'].values
    data1_date = data1_filtered['date'].values
    data1_tumorsize = data1_filtered['tumor_size'].values
    data2_subject = data2_filtered['subject'].values
    data2_tumorsize = data2_filtered['tumor_size'].values

    path = "/mnt/d/research/Liver/"

    A, P, D, label, tumor_size = [], [], [], [], []

    # data1: path / subject / date 폴더
    for subj, date, ts in zip(data1_subject, data1_date, data1_tumorsize):
        folder = os.path.join(path, 'data1correct', str(subj), str(date))
        a_file = os.path.join(folder, "A.nii.gz")
        p_file = os.path.join(folder, "P.nii.gz")
        d_file = os.path.join(folder, "D.nii.gz")
        l_file = os.path.join(folder, "label.nii.gz")

        if all(os.path.exists(f) for f in [a_file, p_file, d_file, l_file]):
            A.append(a_file)
            P.append(p_file)
            D.append(d_file)
            label.append(l_file)
            tumor_size.append(ts)  # ← 추가
        else:
            print(f"[누락] {folder}")

    # data2: path / subject 폴더
    for subj, ts in zip(data2_subject, data2_tumorsize):  # ← ts 추가
        folder = os.path.join(path, 'data2', str(subj))
        a_file = os.path.join(folder, "A.nii.gz")
        p_file = os.path.join(folder, "P.nii.gz")
        d_file = os.path.join(folder, "D.nii.gz")
        l_file = os.path.join(folder, "label.nii.gz")

        if all(os.path.exists(f) for f in [a_file, p_file, d_file, l_file]):
            A.append(a_file)
            P.append(p_file)
            D.append(d_file)
            label.append(l_file)
            tumor_size.append(ts)  # ← 추가
        else:
            print(f"[누락] {folder}")

    print(f"\n총 {len(A)}개 케이스 로드됨")
    print(f"  tumor_size 개수 확인: {len(tumor_size)}개")  # A와 같아야 함

    print(f"\n총 {len(A)}개 케이스 로드됨")
    print(f"  data1: {len(data1_subject)}개 중 매칭")
    print(f"  data2: {len(data2_subject)}개 중 매칭")

    #A[i]

    allsubject, alldate, alltumor_size = [], [], []
    allAP  = [[] for _ in range(6)]  # 0: initial, 1~5: attempt 1~5
    allAD  = [[] for _ in range(6)]
    allPD  = [[] for _ in range(6)]
    allMean= [[] for _ in range(6)]
    
    print(A[:10])
    # A = [
    #     '/mnt/d/research/Liver/data2/3363505/A.nii.gz',
    #     '/mnt/d/research/Liver/data2/7059640/A.nii.gz'
    # ]

    # P = [
    #     '/mnt/d/research/Liver/data2/3363505/P.nii.gz',
    #     '/mnt/d/research/Liver/data2/7059640/P.nii.gz'
    # ]

    # D = [
    #     '/mnt/d/research/Liver/data2/3363505/D.nii.gz',
    #     '/mnt/d/research/Liver/data2/7059640/D.nii.gz'
    # ]
    for i in range(len(A)): # len(A)
        if len(A[i].split('/')) == 9:
            subject = A[i].split("/")[6]
            date = A[i].split("/")[7]
            savepath = os.path.join(f"./ablationregis", subject, date)
            os.makedirs(savepath, exist_ok=True)
            print(f"Start {subject}_{date} files!")

            allsubject.append(subject)
            alldate.append(date)
            alltumor_size.append(tumor_size[i])
        else:
            subject = A[i].split('/')[6]
            savepath = os.path.join(f"./ablationregis", subject)
            os.makedirs(savepath, exist_ok=True)
            print(f"Start {subject} files!")

            allsubject.append(subject)
            alldate.append(0)
            alltumor_size.append(tumor_size[i])

        # 지금 현재 volume들의 dice 구하기
        # 0번째
        Ainitial = A[i]
        Pinitial = P[i]
        Dinitial = D[i]
        
        liverpath = os.path.join(savepath, "register", "1", "liver")
        extract_segmentation(Ainitial, os.path.join(liverpath, "A"))
        extract_segmentation(Pinitial, os.path.join(liverpath, "P"))
        extract_segmentation(Dinitial, os.path.join(liverpath, "D"))

        # liver 읽기
        liverA = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(liverpath, "A", "liver.nii.gz")))
        liverP = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(liverpath, "P", "liver.nii.gz")))
        liverD = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(liverpath, "D", "liver.nii.gz")))

        # sitk.WriteImage(liverA, os.path.join(savepath, "register", "1", "A.nii.gz"))
        # sitk.WriteImage(liverP, os.path.join(savepath, "register", "1", "P.nii.gz"))
        # sitk.WriteImage(liverD, os.path.join(savepath, "register", "1", "D.nii.gz"))
 
        APdice = compute_dice(liverA, liverP)
        ADdice = compute_dice(liverA, liverD)
        PDdice = compute_dice(liverP, liverD)
        meandice = (APdice + ADdice + PDdice) / 3

        print(f"ininitial => {APdice:>.3f} {ADdice:>.3f} {PDdice:>.3f}") 

        allAP[0].append(APdice)
        allAD[0].append(ADdice)
        allPD[0].append(PDdice)
        allMean[0].append(meandice)

        # initial dice가 이미 0.96 초과면 registration 생략
        if meandice > 0.96:
            print(f"No registration needed")
            for r in range(1, 6):
                allAP[r].append(0.0)
                allAD[r].append(0.0)
                allPD[r].append(0.0)
                allMean[r].append(0.0)
            continue   

        # 초기 입력 설정
        current_A = A[i]
        current_P = P[i]
        current_D = D[i]

        for r in range(1, 5):
            register_method = Registration(REGISTRATION_CONFIGS[r+1], r+1)
            os.makedirs(os.path.join(savepath, "register", f"{r+1}", "liver"), exist_ok=True)

            liverpath = os.path.join(savepath, "register", f"{r+1}", "liver")
            print(f"Attempt {r+1}")

            # r=0: 원본 입력, r>0: 이전 attempt 결과 입력
            register_A, register_P, register_D = register_method.run(current_A, current_P, current_D)

            sitk.WriteImage(register_A, os.path.join(savepath, "register", f"{r+1}", "A.nii.gz"))
            sitk.WriteImage(register_P, os.path.join(savepath, "register", f"{r+1}", "P.nii.gz"))
            sitk.WriteImage(register_D, os.path.join(savepath, "register", f"{r+1}", "D.nii.gz"))

            Av = os.path.join(savepath, "register", f"{r+1}", "A.nii.gz")
            Pv = os.path.join(savepath, "register", f"{r+1}", "P.nii.gz")
            Dv = os.path.join(savepath, "register", f"{r+1}", "D.nii.gz")

            # 다음 attempt의 입력으로 현재 결과 설정
            current_A = Av
            current_P = Pv
            current_D = Dv

            extract_segmentation(Av, os.path.join(liverpath, "A"))
            extract_segmentation(Pv, os.path.join(liverpath, "P"))
            extract_segmentation(Dv, os.path.join(liverpath, "D"))

            liverA = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(liverpath, "A", "liver.nii.gz")))
            liverP = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(liverpath, "P", "liver.nii.gz")))
            liverD = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(liverpath, "D", "liver.nii.gz")))

            APdice   = compute_dice(liverA, liverP)
            ADdice   = compute_dice(liverA, liverD)
            PDdice   = compute_dice(liverP, liverD)
            meandice = (APdice + ADdice + PDdice) / 3

            print(f"  {r+1} => {APdice:.3f} {ADdice:.3f} {PDdice:.3f}")

            allAP[r+1].append(APdice)
            allAD[r+1].append(ADdice)
            allPD[r+1].append(PDdice)
            allMean[r+1].append(meandice)

            gc.collect()
            if meandice > 0.96:
                print(f"  → {r+1}번째에서 중단 (mean: {meandice:.3f})")
                for remaining in range(r+2, 6):
                    allAP[remaining].append(0.0)
                    allAD[remaining].append(0.0)
                    allPD[remaining].append(0.0)
                    allMean[remaining].append(0.0)
                break

            """
            register 5번 도는데 1번 돌때마다 mean dice가 0.96이 넘으면 멈추고 
            멈춘 r 다음 r+1인 dice 리스트에는 0.0 집어넣고 다음 데이터로 넘어가기
            """

    df = {}
    df['subject']    = allsubject
    df['date']       = alldate
    df['tumor_size'] = alltumor_size

    for r in range(6):  # 0: initial, 1~5: attempt 1~5
        label = 'initial' if r == 0 else str(r)
        df[f'APdice{label}'] = allAP[r]
        df[f'ADdice{label}'] = allAD[r]
        df[f'PDdice{label}'] = allPD[r]
        df[f'Mean{label}']   = allMean[r]

    # 길이 맞추기
    max_len = max(len(v) for v in df.values())
    data_padded = {k: v + [0.0] * (max_len - len(v)) for k, v in df.items()}

    data = pd.DataFrame(data_padded)
    data.to_excel("./alldata_metrics_2.xlsx", index=False)

if __name__=='__main__':
    main()