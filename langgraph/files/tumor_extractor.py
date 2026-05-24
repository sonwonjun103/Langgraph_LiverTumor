import subprocess
import os
from typing import Optional, List


class TumorExtractor:
    """Extract tumor segmentation using nnUNet"""

    def __init__(self,
                 input_folder: str,
                 output_path: str,
                 dataset_n: str = "003",
                 config: str = "3d_fullres",
                 folds: Optional[List[int]] = None):
        """
        Args:
            input_folder : Patient root folder (liver 폴더 포함)
            output_path  : 결과 저장 루트 (e.g. ./results/1043712)
            dataset_n    : nnUNet dataset number
            config       : nnUNet configuration
            folds        : Fold numbers (None = all folds)
        """
        self.input_folder  = os.path.join(output_path, "liver")   # output_path/liver/
        self.output_folder = os.path.join(output_path, "tumor", 'nnUNet')   # output_path/tumor/
        self.dataset_n     = dataset_n
        self.config        = config
        self.folds         = folds

        os.makedirs(self.output_folder, exist_ok=True)
        self.setup_environment()

    def setup_environment(self):
        nnunet_raw          = "/home/sonwonjun/research/FM/nnUNetv2/nnUNet_raw"
        nnunet_preprocessed = "/home/sonwonjun/research/FM/nnUNetv2/nnUNet_preprocessed"
        nnunet_results      = "/home/sonwonjun/research/FM/nnUNetv2/nnUNet_results"
        os.environ['nnUNet_raw']          = nnunet_raw
        os.environ['nnUNet_preprocessed'] = nnunet_preprocessed
        os.environ['nnUNet_results']      = nnunet_results
        print(f"nnUNet_results set to: {nnunet_results}")

    def build_command(self, save_probabilities: bool = False) -> List[str]:
        cmd = [
            "nnUNetv2_predict",
            "-i", self.input_folder,
            "-o", self.output_folder,
            "-d", self.dataset_n,
            "-c", self.config
        ]
        if self.folds is not None:
            cmd.extend(["-f"] + [str(f) for f in self.folds])
        if save_probabilities:
            cmd.append("--save_probabilities")
        return cmd

    def run(self,
            save_probabilities: bool = False,
            verbose: bool = True) -> bool:
        print("\n" + "=" * 60)
        print("Running nnUNet Tumor Segmentation")
        print("=" * 60)
        print(f"Input folder : {self.input_folder}")
        print(f"Output folder: {self.output_folder}")
        print(f"Dataset      : {self.dataset_n}")
        print(f"Config       : {self.config}")
        print("=" * 60 + "\n")

        if not os.path.exists(self.input_folder):
            print(f"Error: Input folder does not exist: {self.input_folder}")
            return False

        input_files = [f for f in os.listdir(self.input_folder)
                       if f.endswith(('.nii.gz', '.nii'))]
        if not input_files:
            print(f"Error: No NIfTI files found in {self.input_folder}")
            return False

        print(f"Found {len(input_files)} input file(s)")
        cmd = self.build_command(save_probabilities)
        print(f"Command: {' '.join(cmd)}\n")

        try:
            if verbose:
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                for line in process.stdout:
                    print(line, end='')
                return_code = process.wait()
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return_code = result.returncode
                print(result.stdout)

            if return_code == 0:
                print("\n" + "=" * 60)
                print("✓ Tumor segmentation completed successfully")
                print("=" * 60)
                output_files = [f for f in os.listdir(self.output_folder)
                                if f.endswith(('.nii.gz', '.nii'))]
                print(f"Generated {len(output_files)} output file(s)")
                return True
            else:
                print(f"\n✗ nnUNet prediction failed with return code {return_code}")
                return False

        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error: {e.stderr}")
            return False
        except FileNotFoundError:
            print("\n✗ Error: nnUNetv2_predict command not found")
            return False
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")
            return False

    def verify_output(self) -> bool:
        if not os.path.exists(self.output_folder):
            return False
        files = [f for f in os.listdir(self.output_folder)
                 if f.endswith(('.nii.gz', '.nii'))]
        return len(files) > 0


if __name__ == "__main__":
    extractor = TumorExtractor(
        input_folder="/mnt/d/research/Liver/HCC_nifti_crop/1043712",
        output_path="./results/1043712",
        dataset_n="003",
        config="3d_fullres"
    )
    success = extractor.run(save_probabilities=True, verbose=True)
    print("\nSegmentation completed!" if success else "\nSegmentation failed!")