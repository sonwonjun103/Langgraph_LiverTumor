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
import json
import shutil
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
    excel_path: str = "./alldata_metrics.xlsx"
    data_root1: str = DATA_PATH1
    data_root2: str = DATA_PATH2
    results_root: str = "./Results"
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

    # nnUNetv2 settings
    nnunet_input_dir: str = "./nnUNet/nnUNet_raw/Dataset001/imagesTs"
    nnunet_dataset: str = "001"
    nnunet_config: str = "3d_fullres"
    nnunet_save_probabilities: bool = True
    nnunet_raw: Optional[str] = None
    nnunet_preprocessed: Optional[str] = None
    nnunet_results: Optional[str] = None

    def __post_init__(self):
        if self.checkpoints is None:
            suffix = f"_{self.data_type}" if self.data_type else ""
            self.checkpoints = {
                "unet": f"./checkpoints/unet/best_model{suffix}.pt",
                "swinunetr": f"./checkpoints/swinunetr/best_model{suffix}.pt",
                "sam3d_adapter": f"./checkpoints/sam_adapter/best_model{suffix}.pt",
            }



def load_test_dataframe(cfg: PipelineConfig) -> pd.DataFrame:
    data = pd.read_excel(cfg.excel_path)
    data = data.sort_values("Mean1").reset_index(drop=True)
    return data.iloc[:cfg.test_size].copy()


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


def prepare_case_node(state: PipelineState) -> PipelineState:
    cfg = state["cfg"]
    case = state["case"]
    case_result_dir = Path(cfg.results_root) / case["case_id"]
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


def registration_loop_node(state: PipelineState) -> PipelineState:
    cfg = state["cfg"]
    input_dir = Path(state["input_dir"])
    case_result_dir = Path(state["case_result_dir"])
    input_format = state.get("input_format", "nifti")

    attempts = []

    if input_format == "nifti":
        # NIfTI volumes are already available. The initial gate uses the
        # pre-existing liver-cropped files (no TotalSegmentator).
        liver_dir = input_dir / "liver_images"
        liver_files_present = all(
            (liver_dir / f"{p}_liver.nii.gz").exists() for p in ("A", "P", "D")
        )

        initial_record = evaluate_initial_liver_gate(
            volume_dir=input_dir,
            cfg=cfg,
            stage="input_liver_check",
            attempt=0,
            config_name="No registration",
        )
        attempts.append(initial_record)

        if initial_record["liver_dice"]["MeanDice"] >= cfg.liver_dice_threshold:
            state["registration_success"] = True
            state["final_attempt_dir"] = str(input_dir)
            state["final_liver_dice"] = initial_record["liver_dice"]["MeanDice"]
            state["attempts"] = attempts
            return state

        # liver files present  -> can skip attempt 1 (the no-op identity registration)
        # liver files missing  -> start from attempt 1 because we have nothing aligned yet
        start_attempt = 2 if liver_files_present else 1
        registration_input_folder = input_dir
    elif input_format == "dicom":
        # DICOM inputs are converted to NIfTI first, then resampled and registered from attempt 1.
        Resampler(input_folder=str(input_dir), output_path=str(case_result_dir)).run()
        start_attempt = 1
        registration_input_folder = case_result_dir
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")

    for attempt in range(start_attempt, cfg.max_registration_attempts + 1):
        reg_param = REGISTRATION_CONFIGS[attempt]
        attempt_dir = run_registration_attempt(
            cfg=cfg,
            case_result_dir=case_result_dir,
            input_folder=registration_input_folder,
            attempt=attempt,
        )

        attempt_record = evaluate_liver_gate(
            volume_dir=attempt_dir,
            cfg=cfg,
            stage=f"registration_attempt_{attempt}",
            attempt=attempt,
            config_name=reg_param.get("name", f"Attempt {attempt}"),
        )
        attempts.append(attempt_record)

        if attempt_record["liver_dice"]["MeanDice"] >= cfg.liver_dice_threshold:
            state["registration_success"] = True
            state["final_attempt_dir"] = str(attempt_dir)
            state["final_liver_dice"] = attempt_record["liver_dice"]["MeanDice"]
            break

    state["attempts"] = attempts
    if not state.get("registration_success"):
        state["failed_reason"] = "registration_liver_dice_below_threshold"
        if attempts:
            state["final_attempt_dir"] = attempts[-1]["attempt_dir"]
            state["final_liver_dice"] = attempts[-1]["liver_dice"]["MeanDice"]
    return state


def route_after_registration(state: PipelineState) -> str:
    return "tumor" if state.get("registration_success") else "finalize"
def registration_voxelmorph_node(state: PipelineState) -> PipelineState:
    """Register P/D phases to A using a pre-trained VoxelMorph network.

    Single forward pass — no retry loop. The result populates the same state
    keys as registration_loop_node so the rest of the graph (tumor extraction,
    finalize) does not need to know which backend ran.
    """
    cfg = state["cfg"]
    case_result_dir = Path(state["case_result_dir"])
    input_dir = Path(state["input_dir"])

    model_path = getattr(cfg, "voxelmorph_model_path", None)
    if not model_path or not Path(model_path).exists():
        state["registration_success"] = False
        state["failed_reason"] = f"voxelmorph_model_not_found: {model_path}"
        state["attempts"] = []
        return state

    from langgraph.files.registration_voxelmorph import VoxelMorphRegistration

    if state.get("input_format") == "dicom":
        Resampler(input_folder=str(input_dir), output_path=str(case_result_dir)).run()
        register_input = case_result_dir
    else:
        register_input = input_dir

    attempt_dir = VoxelMorphRegistration(
        input_folder=str(register_input),
        output_path=str(case_result_dir),
        model_path=model_path,
        device=cfg.device,
        window=tuple(cfg.window),
        attempt=0,
    ).run()

    record = evaluate_liver_gate(
        volume_dir=Path(attempt_dir),
        cfg=cfg,
        stage="voxelmorph",
        attempt=0,
        config_name="VoxelMorph",
    )

    state["attempts"] = [record]
    passes = record["liver_dice"]["MeanDice"] >= cfg.liver_dice_threshold
    state["registration_success"] = bool(passes)
    state["final_attempt_dir"] = attempt_dir
    state["final_liver_dice"] = record["liver_dice"]["MeanDice"]
    if not passes:
        state["failed_reason"] = "voxelmorph_liver_dice_below_threshold"
    return state


def route_registration_method(state: PipelineState) -> str:
    """Pick the registration backend based on cfg.registration_backend."""
    backend = getattr(state["cfg"], "registration_backend", "simpleitk")
    if backend == "voxelmorph":
        return "registration_voxelmorph"
    return "registration"


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


def tumor_extraction_node(state: PipelineState) -> PipelineState:
    cfg = state["cfg"]
    case = state["case"]
    attempt_dir = Path(state["final_attempt_dir"])
    tumor_root = Path(state["case_result_dir"]) / "tumor"
    gt_path = resample_label_to_reference(
        case["label"],
        attempt_dir / "P.nii.gz",
        Path(state["case_result_dir"]) / "gt_label_resampled.nii.gz",
    )

    available_runners = {
        "unet": lambda out: run_unet(attempt_dir, out, cfg),
        "swinunetr": lambda out: run_swinunetr(attempt_dir, out, cfg),
        "sam3d_adapter": lambda out: run_sam_adapter(attempt_dir, out, cfg, gt_path=gt_path),
        "nnunetv2": lambda out: run_nnunetv2(attempt_dir, out, cfg, case["case_id"]),
    }

    selected = _normalize_model_selection(cfg.models)
    unknown = [name for name in selected if name not in available_runners]
    if unknown:
        raise ValueError(
            f"Unknown tumor models in cfg.models: {unknown}. "
            f"Choose from {SUPPORTED_TUMOR_MODELS}."
        )

    tumor_metrics = {}
    for model_name in selected:
        runner = available_runners[model_name]
        model_dir = tumor_root / model_name
        pred_path = runner(model_dir)
        if pred_path is None:
            tumor_metrics[model_name] = {"status": "failed_or_skipped"}
            write_json(model_dir / "metrics.json", tumor_metrics[model_name])
            continue

        metrics = compute_prediction_metrics(pred_path, gt_path)
        metrics["status"] = "ok"
        tumor_metrics[model_name] = metrics
        write_json(model_dir / "metrics.json", metrics)

    state["tumor_metrics"] = tumor_metrics
    return state


def tumor_extraction_per_attempt_node(state: PipelineState) -> PipelineState:
    """Run tumor extraction for every registration attempt (1..max_registration_attempts).

    Cumulative rule:
      - For each attempt N, check the liver gate.
      - As soon as one attempt's liver mean Dice >= cfg.liver_dice_threshold, every
        subsequent attempt reuses that attempt's registered volumes (no re-registration)
        and runs tumor extraction on the copied volumes.
      - Attempts before the first pass each run their own registration hyperparameter,
        then run tumor extraction regardless of whether they pass the gate.

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

    for attempt in range(1, cfg.max_registration_attempts + 1):
        if passed_source_dir is not None:
            attempt_dir = per_attempt_root / f"attempt_{attempt}_reused"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            for phase in ("A", "P", "D"):
                shutil.copy2(
                    passed_source_dir / f"{phase}.nii.gz",
                    attempt_dir / f"{phase}.nii.gz",
                )
            config_name = "reused (previous attempt passed liver gate)"
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

        gate = evaluate_liver_gate(
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
            pred_path = runners[model_name](model_dir)
            if pred_path is None:
                tumor_metrics[model_name] = {"status": "failed_or_skipped"}
                write_json(model_dir / "metrics.json", tumor_metrics[model_name])
                continue
            metrics = compute_prediction_metrics(pred_path, gt_path)
            metrics["status"] = "ok"
            tumor_metrics[model_name] = metrics
            write_json(model_dir / "metrics.json", metrics)

        per_attempt_results.append({
            "attempt": attempt,
            "attempt_dir": str(attempt_dir),
            "registration_performed": registration_performed,
            "liver_dice": gate["liver_dice"],
            "tumor_metrics": tumor_metrics,
            "config": config_name,
        })

        if (
            passed_source_dir is None
            and gate["liver_dice"]["MeanDice"] >= cfg.liver_dice_threshold
        ):
            passed_source_dir = attempt_dir

    state["per_attempt_results"] = per_attempt_results
    write_json(case_result_dir / "per_attempt_results.json", {"results": per_attempt_results})
    return state


def finalize_node(state: PipelineState) -> PipelineState:
    case_result_dir = Path(state["case_result_dir"])
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
    }
    summary_path = case_result_dir / "summary.json"
    write_json(summary_path, summary)
    state["summary_path"] = str(summary_path)
    return state


def route_after_registration(state: PipelineState) -> str:
    """Pick what to do after the initial registration node.

    - registration_success=True  → "tumor" (single-best inference + per-attempt sweep)
    - registration_success=False → "tumor_per_attempt" (skip single-best, still run the
      cumulative 1..N sweep so we get tumor metrics for every attempt even when the
      liver gate never passes)
    """
    return "tumor" if state.get("registration_success") else "tumor_per_attempt"


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("prepare", prepare_case_node)
    graph.add_node("registration", registration_loop_node)
    graph.add_node("registration_voxelmorph", registration_voxelmorph_node)
    graph.add_node("tumor", tumor_extraction_node)
    graph.add_node("tumor_per_attempt", tumor_extraction_per_attempt_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "prepare")
    graph.add_conditional_edges(
        "prepare",
        route_registration_method,
        {
            "registration": "registration",
            "registration_voxelmorph": "registration_voxelmorph",
        },
    )
    graph.add_conditional_edges(
        "registration",
        route_after_registration,
        {"tumor": "tumor", "tumor_per_attempt": "tumor_per_attempt"},
    )
    graph.add_conditional_edges(
        "registration_voxelmorph",
        route_after_registration,
        {"tumor": "tumor", "tumor_per_attempt": "tumor_per_attempt"},
    )
    graph.add_edge("tumor", "tumor_per_attempt")
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
    parser.add_argument("--excel_path", default="./alldata_metrics.xlsx")
    parser.add_argument("--data_root1", default=None)
    parser.add_argument("--data_root2", default=None)
    parser.add_argument("--input_format", choices=["auto", "nifti", "dicom"], default="auto")
    parser.add_argument("--data_type", choices=["ct", "liver"], default="ct",
                        help="Tumor inference input variant. Must match the data used at training time.")
    parser.add_argument("--liver_dice_threshold", type=float, default=0.95)
    parser.add_argument("--max_registration_attempts", type=int, default=5)
    parser.add_argument("--skip_nnunet_name_check", action="store_true",
                        help="Skip the nnUNet test-case name sanity check at startup.")
    return parser.parse_args()


def build_cfg_from_args(args: argparse.Namespace) -> PipelineConfig:
    kwargs: Dict[str, Any] = dict(
        results_root=args.results_root,
        excel_path=args.excel_path,
        input_format=args.input_format,
        data_type=args.data_type,
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
    Path(cfg.results_root).mkdir(parents=True, exist_ok=True)

    if not args.skip_nnunet_name_check:
        try:
            report = validate_nnunet_test_case_names(cfg)
            write_json(Path(cfg.results_root) / "nnunet_test_case_name_report.json", report)
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
