"""Standalone CLI version of pipeline.ipynb.

Run a single case or the whole test batch through the LangGraph pipeline.

Examples
--------
    python pipeline.py                                 # single mode, first case
    python pipeline.py --mode single --case_id 1309662
    python pipeline.py --mode single --models unet
    python pipeline.py --mode batch
    python pipeline.py --mode batch --registration_backend voxelmorph
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "LangGraph is required for this notebook. Install it in this environment with: pip install langgraph"
    ) from exc

from config import DATA_PATH1, DATA_PATH2
from main import format_path_part
from langgraph.files.config import REGISTRATION_CONFIGS
from langgraph.files.resampler import Resampler
from langgraph.files.registration import Registration
from langgraph.files.inference_unet import run_unet
from langgraph.files.inference_swinunetr import run_swinunetr
from langgraph.files.inference_sam_adapter import run_sam_adapter
from langgraph.files.inference_nnunetv2 import run_nnunetv2

try:
    from externalregis import extract_segmentation as fallback_extract_liver
except Exception as exc:
    fallback_extract_liver = None
    print(f"externalregis liver extraction fallback unavailable: {exc}")


@dataclass
class PipelineConfig:
    sweep_xlsx: str = "./register_dice_sweep.xlsx"
    data_root1: str = DATA_PATH1
    data_root2: str = DATA_PATH2
    results_root: str = "./Results"
    # Per-run timestamp folder; auto-filled in __post_init__ if blank.
    # Final layout: results_root/<data_type>/<timestamp>/<case_id>/
    timestamp: str = ""
    test_size: int = 34
    liver_dice_threshold: float = 0.95
    max_registration_attempts: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Input handling.
    # auto: infer from A/P/D paths. nifti: start from liver Dice check. dicom: resample + registration attempt 1.
    input_format: str = "auto"

    # Tumor inference input variant. Must match how the checkpoints were trained.
    # "ct"    -> source_dir/{A,P,D}.nii.gz                  (original CT phases)
    # "liver" -> source_dir/{AliverAv,PliverPv,DliverDv}.nii.gz
    #           (the liver-cropped files prepared at training time)
    data_type: str = "ct"

    # Registration backend.
    # "simpleitk": iterative multi-attempt SimpleITK registration with retry on the liver gate.
    # "voxelmorph": single forward pass through a pre-trained VoxelMorph network.
    registration_backend: str = "simpleitk"
    voxelmorph_model_path: str = "./checkpoints/voxelmorph/best_model.pt"

    # If True, reuse existing attempt folders on disk: when per_attempt/attempt_N
    # (or attempt_N_reused) already has A/P/D.nii.gz, skip the registration /
    # copy step and re-run only tumor extraction. Useful when new model
    # checkpoints arrive and you just want to refresh their predictions.
    reuse_registration: bool = True

    # Pull cached A/P/D registrations from another run's timestamp folder.
    # "" = no cross-run cache. "latest" = newest sibling timestamp under
    # Results/<data_type>/. Otherwise pass an explicit timestamp string like
    # "20260606_143217".
    cache_from: str = ""

    # Tumor inference selection.
    # Pick a subset like ("unet",) for a single model, or keep all four to run them together.
    # Override at runtime with --models on the CLI.
    models: tuple[str, ...] = ("unet", "swinunetr", "sam3d_adapter", "nnunetv2")

    # Direct model inference settings
    window: tuple[float, float] = (0, 150)
    roi_size: tuple[int, int, int] = (96, 128, 128)
    sw_batch_size: int = 4
    sw_overlap: float = 0.5
    sw_mode: str = "gaussian"
    threshold: float = 0.5

    # Model architecture args must match training.
    channels: tuple[int, ...] = (16, 32, 64, 128, 256)
    strides: tuple[int, ...] = (2, 2, 2, 2)
    num_res_units: int = 2
    img_size: tuple[int, int, int] = (96, 384, 384)
    feature_size: int = 48
    use_checkpoint: bool = False
    sam_patch_size: int = 16
    sam_tubelet_size: int = 16
    sam_embed_dim: int = 128
    sam_depth: int = 4
    sam_num_heads: int = 4
    sam_encoder_channels: int = 128
    sam_decoder_channels: int = 64
    sam_adapter_ratio: float = 0.5

    checkpoints: Dict[str, str] = None

    # nnUNetv2 settings. nnunet_dataset="auto" maps from data_type in __post_init__:
    # data_type="ct" -> Dataset001, data_type="liver" -> Dataset002. Pass an
    # explicit "001"/"002"/... to override.
    nnunet_input_dir: str = "./nnUNet/nnUNet_raw/Dataset001/imagesTs"
    nnunet_dataset: str = "auto"
    nnunet_config: str = "3d_fullres"
    nnunet_save_probabilities: bool = True
    # Mirror main.py's defaults so pipeline.py finds the trained nnUNet weights
    # without requiring nnUNet_raw / _preprocessed / _results to be exported.
    nnunet_raw: Optional[str] = "/home/sonwonjun/research/Liver/finalcode/nnUNet/nnUNet_raw"
    nnunet_preprocessed: Optional[str] = "/home/sonwonjun/research/Liver/finalcode/nnUNet/nnUNet_preprocessed"
    nnunet_results: Optional[str] = "/home/sonwonjun/research/Liver/finalcode/nnUNet/nnUNet_results"

    def __post_init__(self):
        if self.checkpoints is None:
            dt = self.data_type or "ct"

            def _resolve_ckpt(model_subdir: str) -> str:
                """Latest timestamp folder's best_model.pt, with fallback to the
                pre-timestamp ./checkpoints/<model>/<data_type>/best_model.pt path."""
                base = Path(f"./checkpoints/{model_subdir}/{dt}")
                flat = base / "best_model.pt"
                if base.exists():
                    runs = sorted(
                        (p for p in base.iterdir()
                         if p.is_dir() and (p / "best_model.pt").exists()),
                        reverse=True,
                    )
                    if runs:
                        return str(runs[0] / "best_model.pt")
                return str(flat)

            self.checkpoints = {
                "unet": _resolve_ckpt("unet"),
                "swinunetr": _resolve_ckpt("swinunetr"),
                "sam3d_adapter": _resolve_ckpt("sam_adapter"),
            }
        if self.nnunet_dataset == "auto":
            self.nnunet_dataset = "002" if self.data_type == "liver" else "001"
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")



def load_test_dataframe(cfg: PipelineConfig) -> pd.DataFrame:
    # Test set comes from the lowest-mean1 cases in register_dice_sweep.xlsx.
    from dataset_split import split_train_test

    _, test_df = split_train_test(
        sweep_xlsx=cfg.sweep_xlsx,
        test_size=cfg.test_size,
    )
    return test_df


def get_case_from_row(row: pd.Series, cfg: PipelineConfig) -> Dict[str, Any]:
    subject = format_path_part(row["subject"])
    date = format_path_part(row["date"])
    if date == "0":
        source_dir = Path(cfg.data_root2) / subject
        case_id = subject
    else:
        source_dir = Path(cfg.data_root1) / subject / date
        case_id = f"{subject}_{date}"

    return {
        "subject": subject,
        "date": date,
        "case_id": case_id,
        "source_dir": str(source_dir),
        # CT phases (registration + ct-mode tumor input)
        "A": str(source_dir / "A.nii.gz"),
        "P": str(source_dir / "P.nii.gz"),
        "D": str(source_dir / "D.nii.gz"),
        # Liver-cropped phases produced at training time (initial liver gate +
        # liver-mode tumor input when no registration runs)
        "A_liver": str(source_dir / "AliverAv.nii.gz"),
        "P_liver": str(source_dir / "PliverPv.nii.gz"),
        "D_liver": str(source_dir / "DliverDv.nii.gz"),
        "label": str(source_dir / "label.nii.gz"),
    }


def expected_test_case_ids(cfg: PipelineConfig) -> List[str]:
    return [get_case_from_row(row, cfg)["case_id"] for _, row in load_test_dataframe(cfg).iterrows()]


def get_test_cases(cfg: PipelineConfig) -> List[Dict[str, Any]]:
    df = load_test_dataframe(cfg)
    cases = []
    for _, row in df.iterrows():
        case = get_case_from_row(row, cfg)
        required = [case["A"], case["P"], case["D"], case["label"]]
        if all(Path(p).exists() for p in required):
            cases.append(case)
        else:
            print(f"skip missing case: {case['case_id']}")
    return cases


def is_nifti_path(path: str | Path) -> bool:
    path = Path(path)
    name = path.name.lower()
    return path.is_file() and (name.endswith(".nii") or name.endswith(".nii.gz"))


def contains_dicom_files(path: str | Path) -> bool:
    path = Path(path)
    if path.is_file():
        suffix = path.suffix.lower()
        return suffix in {".dcm", ".ima"}
    if not path.is_dir():
        return False
    return any(p.suffix.lower() in {".dcm", ".ima"} for p in path.rglob("*"))


def infer_input_format(case: Dict[str, Any], cfg: PipelineConfig) -> str:
    if cfg.input_format != "auto":
        return cfg.input_format

    phase_paths = [case[phase] for phase in ("A", "P", "D")]
    if all(is_nifti_path(path) for path in phase_paths):
        return "nifti"
    if all(contains_dicom_files(path) for path in phase_paths):
        return "dicom"
    raise ValueError(
        f"Could not infer input format for {case['case_id']}. "
        "Expected all A/P/D inputs to be NIfTI files or all to be DICOM directories/files."
    )


def dicom_to_nifti(dicom_path: str | Path, output_path: str | Path) -> str:
    dicom_path = Path(dicom_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dicom_path.is_file():
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(dicom_path))
        image = reader.Execute()
    else:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_path))
        if not series_ids:
            raise FileNotFoundError(f"No DICOM series found in {dicom_path}")
        series_files = [
            sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_path), series_id)
            for series_id in series_ids
        ]
        dicom_files = max(series_files, key=len)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_files)
        image = reader.Execute()

    sitk.WriteImage(image, str(output_path))
    return str(output_path)


def prepare_phase_inputs(case: Dict[str, Any], case_result_dir: Path, cfg: PipelineConfig) -> tuple[Path, str]:
    """Stage both the CT phases and the training-side liver-cropped phases.

    Layout after this step:
      input_dir/{A,P,D}.nii.gz                        # CT (used for registration)
      input_dir/liver_images/{A,P,D}_liver.nii.gz     # liver-cropped (used for the
                                                       # initial liver gate and as
                                                       # the model input when
                                                       # cfg.data_type == "liver")
    """
    input_format = infer_input_format(case, cfg)

    if input_format == "nifti":
        input_dir = case_result_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        for phase in ("A", "P", "D"):
            shutil.copy2(case[phase], input_dir / f"{phase}.nii.gz")
        liver_dir = input_dir / "liver_images"
        liver_dir.mkdir(parents=True, exist_ok=True)
        for phase in ("A", "P", "D"):
            liver_src = case.get(f"{phase}_liver")
            if liver_src and Path(liver_src).exists():
                shutil.copy2(liver_src, liver_dir / f"{phase}_liver.nii.gz")
        return input_dir, input_format

    if input_format == "dicom":
        input_dir = case_result_dir / "dicom_nifti"
        input_dir.mkdir(parents=True, exist_ok=True)
        for phase in ("A", "P", "D"):
            dicom_to_nifti(case[phase], input_dir / f"{phase}.nii.gz")
        return input_dir, input_format

    raise ValueError(f"Unsupported input_format: {input_format}")


def nnunet_images_ts_case_ids(images_ts_dir: str | Path) -> Dict[str, Any]:
    images_ts_dir = Path(images_ts_dir)
    if not images_ts_dir.exists():
        return {"exists": False, "path": str(images_ts_dir), "case_ids": [], "bad_files": [], "channel_counts": {}}

    case_to_channels: Dict[str, List[str]] = {}
    bad_files = []
    for path in sorted(images_ts_dir.glob("*.nii.gz")):
        stem = path.name[:-7]
        if "_" not in stem:
            bad_files.append(path.name)
            continue
        case_id, channel = stem.rsplit("_", 1)
        case_to_channels.setdefault(case_id, []).append(channel)

    return {
        "exists": True,
        "path": str(images_ts_dir),
        "case_ids": sorted(case_to_channels),
        "bad_files": bad_files,
        "channel_counts": {case_id: len(channels) for case_id, channels in sorted(case_to_channels.items())},
        "non_3_channel_cases": {
            case_id: channels
            for case_id, channels in sorted(case_to_channels.items())
            if len(channels) != 3
        },
    }


def validate_nnunet_test_case_names(cfg: PipelineConfig) -> Dict[str, Any]:
    expected = expected_test_case_ids(cfg)
    dataset_reports = {}
    for dataset_name in ("Dataset001", "Dataset002"):
        images_ts = Path("./nnUNet/nnUNet_raw") / dataset_name / "imagesTs"
        report = nnunet_images_ts_case_ids(images_ts)
        actual = report["case_ids"]
        report["same_set_as_excel_test_split"] = set(actual) == set(expected)
        report["same_order_as_excel_test_split"] = actual == expected
        report["missing_from_nnunet"] = sorted(set(expected) - set(actual))
        report["extra_in_nnunet"] = sorted(set(actual) - set(expected))
        dataset_reports[dataset_name] = report

    return {
        "expected_test_cases": expected,
        "n_expected": len(expected),
        "datasets": dataset_reports,
    }


def write_json(path: str | Path, data: Dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def binary_array(path: str | Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float((2 * inter + eps) / (denom + eps))


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))


def resample_label_to_reference(label_path: str | Path, reference_path: str | Path, output_path: str | Path) -> str:
    label_img = sitk.ReadImage(str(label_path))
    ref_img = sitk.ReadImage(str(reference_path))
    resampled = sitk.Resample(
        label_img,
        ref_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        label_img.GetPixelID(),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled, str(output_path))
    return str(output_path)


def compute_prediction_metrics(pred_path: str | Path, gt_path: str | Path) -> Dict[str, float]:
    pred = binary_array(pred_path)
    gt = binary_array(gt_path)
    return {"dice": dice_score(pred, gt), "iou": iou_score(pred, gt)}


def extract_liver_masks(attempt_dir: Path) -> Dict[str, str]:
    try:
        from langgraph.files.liver_extractor import LiverExtractor  # type: ignore

        extractor = LiverExtractor(input_folder=str(attempt_dir), output_path=str(attempt_dir))
        result = extractor.run_all()
        if isinstance(result, dict) and all(phase in result for phase in ("A", "P", "D")):
            return result
    except Exception as exc:
        print(f"LiverExtractor unavailable or failed, trying fallback: {exc}")

    if fallback_extract_liver is None:
        raise RuntimeError(
            "No usable liver extractor found. Install TotalSegmentator or provide "
            "externalregis.extract_segmentation."
        )

    liver_root = attempt_dir / "liver_masks"
    mask_paths = {}
    for phase in ("A", "P", "D"):
        phase_out = liver_root / phase
        phase_out.mkdir(parents=True, exist_ok=True)
        fallback_extract_liver(str(attempt_dir / f"{phase}.nii.gz"), str(phase_out))
        mask_paths[phase] = str(phase_out / "liver.nii.gz")

    shutil.copy2(mask_paths["P"], attempt_dir / "liver.nii.gz")
    return mask_paths


def compute_liver_dice(mask_paths: Dict[str, str]) -> Dict[str, float]:
    liver_a = binary_array(mask_paths["A"])
    liver_p = binary_array(mask_paths["P"])
    liver_d = binary_array(mask_paths["D"])
    ap = dice_score(liver_a, liver_p)
    ad = dice_score(liver_a, liver_d)
    pd_ = dice_score(liver_p, liver_d)
    return {"APdice": ap, "ADdice": ad, "PDdice": pd_, "MeanDice": float((ap + ad + pd_) / 3)}


class PipelineState(TypedDict, total=False):
    cfg: PipelineConfig
    case: Dict[str, Any]
    case_result_dir: str
    input_dir: str
    input_format: str
    attempts: List[Dict[str, Any]]
    registration_success: bool
    final_attempt_dir: Optional[str]
    final_liver_dice: Optional[float]
    tumor_metrics: Dict[str, Dict[str, float]]
    per_attempt_results: List[Dict[str, Any]]
    failed_reason: Optional[str]
    summary_path: Optional[str]
    start_time: float
    elapsed_seconds: float


def prepare_case_node(state: PipelineState) -> PipelineState:
    cfg = state["cfg"]
    case = state["case"]
    # Layer Results by data_type AND a per-run timestamp so each pipeline
    # invocation is isolated. Final layout:
    #   ./Results/<data_type>/<timestamp>/<case_id>/
    case_result_dir = Path(cfg.results_root) / cfg.data_type / cfg.timestamp / case["case_id"]
    case_result_dir.mkdir(parents=True, exist_ok=True)
    input_dir, input_format = prepare_phase_inputs(case, case_result_dir, cfg)
    write_json(case_result_dir / "case.json", {**case, "input_format": input_format})
    state.update({
        "case_result_dir": str(case_result_dir),
        "input_dir": str(input_dir),
        "input_format": input_format,
        "attempts": [],
        "registration_success": False,
        "final_attempt_dir": None,
        "final_liver_dice": None,
        "tumor_metrics": {},
        "per_attempt_results": [],
        "failed_reason": None,
        "start_time": time.time(),
    })
    return state


def evaluate_liver_gate(volume_dir: Path, cfg: PipelineConfig, stage: str, attempt: int, config_name: str) -> Dict[str, Any]:
    mask_paths = extract_liver_masks(volume_dir)
    dice_info = compute_liver_dice(mask_paths)
    dice_info["attempt"] = attempt
    dice_info["stage"] = stage
    dice_info["threshold"] = cfg.liver_dice_threshold
    write_json(volume_dir / "liver_dice.json", dice_info)
    return {
        "attempt": attempt,
        "attempt_dir": str(volume_dir),
        "stage": stage,
        "registration_performed": stage.startswith("registration"),
        "liver_dice": dice_info,
        "config": config_name,
    }


def evaluate_initial_liver_gate(volume_dir: Path, cfg: PipelineConfig, stage: str, attempt: int, config_name: str) -> Dict[str, Any]:
    """Initial liver gate that uses pre-existing liver-cropped files instead of
    running TotalSegmentator. Expects volume_dir/liver_images/{A,P,D}_liver.nii.gz.
    Falls back to ``evaluate_liver_gate`` (TotalSegmentator) when any of the
    liver files is missing.
    """
    liver_dir = Path(volume_dir) / "liver_images"
    mask_paths = {
        "A": str(liver_dir / "A_liver.nii.gz"),
        "P": str(liver_dir / "P_liver.nii.gz"),
        "D": str(liver_dir / "D_liver.nii.gz"),
    }
    if not all(Path(p).exists() for p in mask_paths.values()):
        return evaluate_liver_gate(volume_dir, cfg, stage, attempt, config_name)

    dice_info = compute_liver_dice(mask_paths)
    dice_info["attempt"] = attempt
    dice_info["stage"] = stage
    dice_info["threshold"] = cfg.liver_dice_threshold
    write_json(Path(volume_dir) / "liver_dice.json", dice_info)
    return {
        "attempt": attempt,
        "attempt_dir": str(volume_dir),
        "stage": stage,
        "registration_performed": False,
        "liver_dice": dice_info,
        "config": config_name,
    }


def run_registration_attempt(
    cfg: PipelineConfig,
    case_result_dir: Path,
    input_folder: Path,
    attempt: int,
) -> Path:
    reg_param = REGISTRATION_CONFIGS[attempt]
    attempt_dir = Registration(
        regis_param=reg_param,
        input_folder=str(input_folder),
        output_path=str(case_result_dir),
        attempt=attempt - 1,
    ).run()
    return Path(attempt_dir)


SUPPORTED_TUMOR_MODELS = ("unet", "swinunetr", "sam3d_adapter", "nnunetv2")


def _normalize_model_selection(value) -> tuple:
    """Accept cfg.models as a string, list, tuple, or None and return a tuple of names.

    Plain strings like ``"swinunetr"`` are treated as a single-model selection
    (instead of being iterated character-by-character).
    """
    if value is None:
        return SUPPORTED_TUMOR_MODELS
    if isinstance(value, str):
        return (value,)
    return tuple(value)




def tumor_extraction_per_attempt_node(state: PipelineState) -> PipelineState:
    """Run tumor extraction for every registration attempt (1..max_registration_attempts).

    Cumulative rule:
      - Attempt 1 (NIfTI input): copy source CT (A/P/D.nii.gz) and the
        training-side liver-cropped files (AliverAv/PliverPv/DliverDv -> liver_images/)
        directly into attempt_1/. NO new registration, NO TotalSegmentator.
        Liver Dice is computed straight from those liver files. This matches
        register_dice_sweep's mean1 definition.
      - Attempt 2..5: if the previous gate already passed, copy the passed
        attempt's volumes into attempt_N_reused/ (no re-registration). Otherwise
        run REGISTRATION_CONFIGS[N] from the case input + TotalSegmentator gate.
      - Tumor extraction always runs at every attempt.

    Results are written under {case_result_dir}/per_attempt/attempt_{N}/.
    """
    cfg = state["cfg"]
    case = state["case"]
    case_result_dir = Path(state["case_result_dir"])
    input_format = state.get("input_format", "nifti")
    per_attempt_root = case_result_dir / "per_attempt"
    per_attempt_root.mkdir(parents=True, exist_ok=True)

    if input_format == "nifti":
        registration_input_folder = Path(state["input_dir"])
    elif input_format == "dicom":
        dicom_resample_dir = case_result_dir / "per_attempt_resample"
        dicom_resample_dir.mkdir(parents=True, exist_ok=True)
        Resampler(input_folder=str(state["input_dir"]), output_path=str(dicom_resample_dir)).run()
        registration_input_folder = dicom_resample_dir
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")

    per_attempt_results: List[Dict[str, Any]] = []
    passed_source_dir: Optional[Path] = None

    reuse = bool(getattr(cfg, "reuse_registration", True))

    # Build the list of per_attempt roots to scan for a cache hit, in
    # priority order: this run first, then optional cross-run sources.
    cache_roots: List[Path] = [per_attempt_root]
    cache_from = (getattr(cfg, "cache_from", "") or "").strip()
    if cache_from:
        data_type_dir = Path(cfg.results_root) / cfg.data_type
        if cache_from.lower() == "latest" and data_type_dir.exists():
            siblings = sorted(
                (p for p in data_type_dir.iterdir()
                 if p.is_dir() and p.name != cfg.timestamp),
                reverse=True,
            )
            for sib in siblings:
                extra = sib / case["case_id"] / "per_attempt"
                if extra.exists():
                    cache_roots.append(extra)
                    break
        else:
            extra = data_type_dir / cache_from / case["case_id"] / "per_attempt"
            if extra.exists():
                cache_roots.append(extra)

    def _link_or_copy(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(os.path.relpath(src, dst.parent), dst)
        except OSError:
            shutil.copy2(src, dst)

    def _stage_cache_into_run(src_dir: Path, dst_dir: Path) -> None:
        """Bring a cached attempt folder into the current run dir (symlink first,
        copy fallback) so tumor outputs land inside this run's timestamp folder."""
        dst_dir.mkdir(parents=True, exist_ok=True)
        for phase in ("A", "P", "D"):
            _link_or_copy(src_dir / f"{phase}.nii.gz", dst_dir / f"{phase}.nii.gz")
        src_liver = src_dir / "liver_images"
        if src_liver.exists():
            for f in src_liver.glob("*_liver.nii.gz"):
                _link_or_copy(f, dst_dir / "liver_images" / f.name)
        src_masks = src_dir / "liver_masks"
        if src_masks.exists():
            for phase_dir in src_masks.iterdir():
                if not phase_dir.is_dir():
                    continue
                for f in phase_dir.iterdir():
                    if f.is_file():
                        _link_or_copy(f, dst_dir / "liver_masks" / phase_dir.name / f.name)
        for fname in ("liver.nii.gz", "liver_dice.json"):
            sf = src_dir / fname
            if sf.exists():
                _link_or_copy(sf, dst_dir / fname)

    def _cached_attempt_dir(att: int) -> Optional[Path]:
        """Return an attempt_<N>[_reused] folder (staged into this run) if
        any cache root has A/P/D ready."""
        for root in cache_roots:
            for name in (f"attempt_{att}", f"attempt_{att}_reused"):
                d = root / name
                if d.exists() and all((d / f"{p}.nii.gz").exists() for p in ("A", "P", "D")):
                    if root == per_attempt_root:
                        return d
                    # Cross-run cache hit: stage into this run dir first.
                    staged = per_attempt_root / name
                    _stage_cache_into_run(d, staged)
                    return staged
        return None

    for attempt in range(1, cfg.max_registration_attempts + 1):
        a_start = time.time()
        cached = _cached_attempt_dir(attempt) if reuse else None
        if cached is not None:
            # Registered volumes already on disk — skip register / copy
            # entirely and just re-run liver gate + tumor extraction.
            attempt_dir = cached
            config_name = f"cached ({attempt_dir.name})"
            registration_performed = False
        elif passed_source_dir is not None:
            # Previous attempt cleared the gate; just clone it.
            attempt_dir = per_attempt_root / f"attempt_{attempt}_reused"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            for phase in ("A", "P", "D"):
                shutil.copy2(
                    passed_source_dir / f"{phase}.nii.gz",
                    attempt_dir / f"{phase}.nii.gz",
                )
            src_liver = passed_source_dir / "liver_images"
            if src_liver.exists():
                dst_liver = attempt_dir / "liver_images"
                dst_liver.mkdir(parents=True, exist_ok=True)
                for f in src_liver.glob("*_liver.nii.gz"):
                    shutil.copy2(f, dst_liver / f.name)
            config_name = "reused (previous attempt passed liver gate)"
            registration_performed = False
        elif attempt == 1 and input_format == "nifti":
            # Attempt 1 for NIfTI input = the training-side liver-cropped files
            # straight from the source directory. No registration, no TotalSegmentator.
            attempt_dir = per_attempt_root / "attempt_1"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            for phase in ("A", "P", "D"):
                shutil.copy2(case[phase], attempt_dir / f"{phase}.nii.gz")
            liver_dir = attempt_dir / "liver_images"
            liver_dir.mkdir(parents=True, exist_ok=True)
            for phase in ("A", "P", "D"):
                liver_src = case.get(f"{phase}_liver")
                if liver_src and Path(liver_src).exists():
                    shutil.copy2(liver_src, liver_dir / f"{phase}_liver.nii.gz")
            config_name = "source files (no register)"
            registration_performed = False
        else:
            attempt_dir = run_registration_attempt(
                cfg=cfg,
                case_result_dir=per_attempt_root,
                input_folder=registration_input_folder,
                attempt=attempt,
            )
            config_name = REGISTRATION_CONFIGS[attempt].get("name", f"Attempt {attempt}")
            registration_performed = True

        # When no registration ran this attempt (attempt_1 source files OR a
        # reused/copied attempt after a prior pass), liver_images/ is already
        # on disk — score the gate directly without re-running TotalSegmentator.
        gate_fn = evaluate_liver_gate if registration_performed else evaluate_initial_liver_gate
        gate = gate_fn(
            volume_dir=attempt_dir,
            cfg=cfg,
            stage=f"per_attempt_{attempt}",
            attempt=attempt,
            config_name=config_name,
        )

        gt_path = resample_label_to_reference(
            case["label"],
            attempt_dir / "P.nii.gz",
            attempt_dir / "gt_label_resampled.nii.gz",
        )

        runners = {
            "unet": lambda out: run_unet(attempt_dir, out, cfg),
            "swinunetr": lambda out: run_swinunetr(attempt_dir, out, cfg),
            "sam3d_adapter": lambda out: run_sam_adapter(attempt_dir, out, cfg, gt_path=gt_path),
            "nnunetv2": lambda out: run_nnunetv2(attempt_dir, out, cfg, case["case_id"]),
        }
        selected = _normalize_model_selection(cfg.models)
        unknown = [name for name in selected if name not in runners]
        if unknown:
            raise ValueError(
                f"Unknown tumor models in cfg.models: {unknown}. "
                f"Choose from {SUPPORTED_TUMOR_MODELS}."
            )
        tumor_metrics: Dict[str, Dict[str, float]] = {}
        tumor_root = attempt_dir / "tumor"
        for model_name in selected:
            model_dir = tumor_root / model_name
            m_t0 = time.time()
            pred_path = runners[model_name](model_dir)
            m_elapsed = round(time.time() - m_t0, 2)
            if pred_path is None:
                tumor_metrics[model_name] = {
                    "status": "failed_or_skipped",
                    "elapsed_seconds": m_elapsed,
                }
                write_json(model_dir / "metrics.json", tumor_metrics[model_name])
                continue
            metrics = compute_prediction_metrics(pred_path, gt_path)
            metrics["status"] = "ok"
            metrics["elapsed_seconds"] = m_elapsed
            tumor_metrics[model_name] = metrics
            write_json(model_dir / "metrics.json", metrics)

        attempt_elapsed = round(time.time() - a_start, 2)
        per_attempt_results.append({
            "attempt": attempt,
            "attempt_dir": str(attempt_dir),
            "registration_performed": registration_performed,
            "liver_dice": gate["liver_dice"],
            "tumor_metrics": tumor_metrics,
            "config": config_name,
            "elapsed_seconds": attempt_elapsed,
            "elapsed_min": round(attempt_elapsed / 60.0, 3),
        })

        if (
            passed_source_dir is None
            and gate["liver_dice"]["MeanDice"] >= cfg.liver_dice_threshold
        ):
            passed_source_dir = attempt_dir

    state["per_attempt_results"] = per_attempt_results
    state["attempts"] = per_attempt_results

    # Pick the first attempt that cleared the liver gate as "final" so
    # finalize_node / case.json / summary.json have something to record.
    threshold = cfg.liver_dice_threshold
    first_passed = next(
        (e for e in per_attempt_results
         if e.get("liver_dice", {}).get("MeanDice", 0.0) >= threshold),
        None,
    )
    if first_passed is not None:
        state["registration_success"] = True
        state["final_attempt_dir"] = first_passed["attempt_dir"]
        state["final_liver_dice"] = first_passed["liver_dice"]["MeanDice"]
        state["tumor_metrics"] = first_passed.get("tumor_metrics", {})
    elif per_attempt_results:
        state["registration_success"] = False
        state["failed_reason"] = "registration_liver_dice_below_threshold"
        last = per_attempt_results[-1]
        state["final_attempt_dir"] = last["attempt_dir"]
        state["final_liver_dice"] = last["liver_dice"]["MeanDice"]
        state["tumor_metrics"] = last.get("tumor_metrics", {})
    else:
        state["registration_success"] = False
        state["failed_reason"] = "no_attempts_completed"
        state["tumor_metrics"] = {}

    write_json(case_result_dir / "per_attempt_results.json", {"results": per_attempt_results})
    return state


def finalize_node(state: PipelineState) -> PipelineState:
    case_result_dir = Path(state["case_result_dir"])
    elapsed = time.time() - state.get("start_time", time.time())
    state["elapsed_seconds"] = elapsed

    # Stamp case.json with the registration outcome so it is visible without
    # opening summary.json.
    registration_status = "ok" if state.get("registration_success") else "all_registration_fail"
    case_info = {
        **state["case"],
        "input_format": state.get("input_format"),
        "registration_status": registration_status,
        "final_liver_dice": state.get("final_liver_dice"),
    }
    write_json(case_result_dir / "case.json", case_info)

    summary = {
        "case": state["case"],
        "input_format": state.get("input_format"),
        "registration_backend": getattr(state["cfg"], "registration_backend", "simpleitk"),
        "registration_success": state.get("registration_success", False),
        "failed_reason": state.get("failed_reason"),
        "registration_attempts": len(state.get("attempts", [])),
        "final_attempt_dir": state.get("final_attempt_dir"),
        "final_liver_dice": state.get("final_liver_dice"),
        "attempts": state.get("attempts", []),
        "tumor_metrics": state.get("tumor_metrics", {}),
        "per_attempt_results": state.get("per_attempt_results", []),
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_min": round(elapsed / 60.0, 3),
    }
    summary_path = case_result_dir / "summary.json"
    write_json(summary_path, summary)
    state["summary_path"] = str(summary_path)
    return state


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("prepare", prepare_case_node)
    graph.add_node("tumor_per_attempt", tumor_extraction_per_attempt_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "tumor_per_attempt")
    graph.add_edge("tumor_per_attempt", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph liver tumor pipeline.")
    parser.add_argument("--mode", choices=["single", "batch"], default="single")
    parser.add_argument("--case_id", default=None,
                        help="Single mode: case id to run. Default = first available test case.")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Tumor models to run. Default: all four "
                             "(unet swinunetr sam3d_adapter nnunetv2).")
    parser.add_argument("--registration_backend", choices=["simpleitk", "voxelmorph"],
                        default="simpleitk")
    parser.add_argument("--voxelmorph_model_path", default=None)
    parser.add_argument("--results_root", default="./Results")
    parser.add_argument("--sweep_xlsx", default="./register_dice_sweep.xlsx",
                        help="register_dice_sweep output. Top test_size cases (mean1 ascending) "
                             "become the test set; the rest are train.")
    parser.add_argument("--data_root1", default=None)
    parser.add_argument("--data_root2", default=None)
    parser.add_argument("--input_format", choices=["auto", "nifti", "dicom"], default="auto")
    parser.add_argument("--data_type", choices=["ct", "liver"], default="ct",
                        help="Tumor inference input variant. Must match the data used at training time.")
    parser.add_argument("--reuse-registration", dest="reuse_registration", action="store_true",
                        default=True,
                        help="(default) Reuse existing per_attempt/attempt_N/A/P/D.nii.gz when "
                             "present; only re-run tumor extraction.")
    parser.add_argument("--no-reuse-registration", dest="reuse_registration", action="store_false",
                        help="Force every attempt to re-run registration even if cached.")
    parser.add_argument("--cache_from", default="",
                        help='Cross-run cache. Pass a timestamp like "20260606_143217" or '
                             '"latest" to scan Results/<data_type>/<that_ts>/<case_id>/per_attempt '
                             'for A/P/D and stage them into this run.')
    parser.add_argument("--nnunet_dataset", default="auto",
                        help='nnUNet dataset id ("001"/"002"/...). '
                             '"auto" picks "001" for data_type=ct and "002" for data_type=liver.')
    parser.add_argument("--nnunet_config", default="3d_fullres")
    parser.add_argument("--liver_dice_threshold", type=float, default=0.95)
    parser.add_argument("--max_registration_attempts", type=int, default=5)
    parser.add_argument("--skip_nnunet_name_check", action="store_true",
                        help="Skip the nnUNet test-case name sanity check at startup.")
    return parser.parse_args()


def build_cfg_from_args(args: argparse.Namespace) -> PipelineConfig:
    kwargs: Dict[str, Any] = dict(
        results_root=args.results_root,
        sweep_xlsx=args.sweep_xlsx,
        input_format=args.input_format,
        data_type=args.data_type,
        nnunet_dataset=args.nnunet_dataset,
        nnunet_config=args.nnunet_config,
        reuse_registration=args.reuse_registration,
        cache_from=args.cache_from,
        registration_backend=args.registration_backend,
        liver_dice_threshold=args.liver_dice_threshold,
        max_registration_attempts=args.max_registration_attempts,
    )
    if args.data_root1 is not None:
        kwargs["data_root1"] = args.data_root1
    if args.data_root2 is not None:
        kwargs["data_root2"] = args.data_root2
    if args.models:
        kwargs["models"] = tuple(args.models)
    if args.voxelmorph_model_path is not None:
        kwargs["voxelmorph_model_path"] = args.voxelmorph_model_path
    return PipelineConfig(**kwargs)


def run_single(cfg: PipelineConfig, pipeline_graph, case_id: Optional[str] = None) -> Dict[str, Any]:
    test_cases = get_test_cases(cfg)
    if not test_cases:
        raise RuntimeError("No test cases found.")
    if case_id is None:
        target = test_cases[0]
    else:
        target = next((c for c in test_cases if c["case_id"] == case_id), None)
        if target is None:
            raise RuntimeError(f"Case id {case_id!r} not found in test cases.")
    print(f"=== Single mode: {target['case_id']} ===")
    result = pipeline_graph.invoke({"cfg": cfg, "case": target})
    print(f"summary: {result['summary_path']}")
    return result


def run_batch(cfg: PipelineConfig, pipeline_graph) -> List[str]:
    cases = get_test_cases(cfg)
    print(f"=== Batch mode: {len(cases)} case(s) ===")
    summaries: List[str] = []
    for case in cases:
        print(f"\n=== Running {case['case_id']} ===")
        result = pipeline_graph.invoke({"cfg": cfg, "case": case})
        summaries.append(result["summary_path"])
    write_json(Path(cfg.results_root) / "batch_summary.json", {"summaries": summaries})
    return summaries


def main() -> None:
    args = parse_args()
    cfg = build_cfg_from_args(args)
    run_dir = Path(cfg.results_root) / cfg.data_type / cfg.timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # Drop a run_metadata.json beside the case folders so this run's settings
    # are recoverable from disk alone (mirrors the train-side metadata).
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        git_commit = None
    cfg_dict = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    cfg_dict["models"] = list(cfg.models) if cfg.models else None
    cfg_dict["window"] = list(cfg.window)
    cfg_dict["roi_size"] = list(cfg.roi_size)
    write_json(run_dir / "run_metadata.json", {
        "timestamp": cfg.timestamp,
        "mode": args.mode,
        "case_id": args.case_id,
        "git_commit": git_commit,
        "cfg": cfg_dict,
    })

    if not args.skip_nnunet_name_check:
        try:
            report = validate_nnunet_test_case_names(cfg)
            write_json(run_dir / "nnunet_test_case_name_report.json", report)
            for dataset_name, ds in report["datasets"].items():
                print(
                    dataset_name,
                    "same_set=", ds["same_set_as_excel_test_split"],
                    "non_3_channel_cases=", len(ds["non_3_channel_cases"]),
                    "missing=", ds["missing_from_nnunet"],
                    "extra=", ds["extra_in_nnunet"],
                )
        except Exception as exc:
            print(f"nnUNet name check skipped: {exc}")

    pipeline_graph = build_graph()

    if args.mode == "single":
        run_single(cfg, pipeline_graph, case_id=args.case_id)
    else:
        run_batch(cfg, pipeline_graph)


if __name__ == "__main__":
    main()
