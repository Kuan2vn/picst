#!/usr/bin/env python3
"""Profile UIIS -> SUIM SFUDA efficiency and optional reliability metrics.

The profiler keeps the adaptation step faithful by launching the repository's
existing train entry points for a short run, while static/inference metrics are
measured in-process from the same model definition.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.backends import cudnn

from core.configs.defaults import _C as DEFAULT_CFG
from core.configs.underwater import add_underwater_label_args, apply_underwater_label_args
from core.datasets import build_dataset
from core.models import build_classifier, build_feature_extractor
from core.utils.misc import intersectionAndUnionGPU, mkdir, strip_prefix_if_present


BASELINE_METHODS = {
    "cal_sfda",
    "tt_sfuda",
    "crossmatch",
    "crots",
    "iapc",
    "sfmim",
    "uotsl",
}

METHOD_LABELS = {
    "tt_sfuda": "TT-SFUDA",
    "cal_sfda": "Cal-SFDA",
    "sfmim": "SFMIM",
    "crossmatch": "CrossMatch",
    "uotsl": "UOTSL",
    "crots": "CROTS",
    "iapc": "IAPC",
    "snd": "SND",
    "remask": "ReMask",
    "remask_dtst": "ReMask+DTST",
}

TRAIN_COMPLEXITY_META = {
    "tt_sfuda": {
        "train_extra_networks": "ema teacher",
        "train_forward_passes_est": "3 after phase-1, 2 in phase-1",
    },
    "cal_sfda": {
        "train_extra_networks": "ema teacher",
        "train_forward_passes_est": "2",
    },
    "crossmatch": {
        "train_extra_networks": "ema teacher",
        "train_forward_passes_est": "3",
    },
    "crots": {
        "train_extra_networks": "ema teacher",
        "train_forward_passes_est": "2",
    },
    "iapc": {
        "train_extra_networks": "ema teacher + frozen source model + prototype bank",
        "train_forward_passes_est": "4",
    },
    "sfmim": {
        "train_extra_networks": "ema teacher + frozen source model",
        "train_forward_passes_est": "multi-scale teacher + source + whole/patch/channel student",
    },
    "uotsl": {
        "train_extra_networks": "ema teacher",
        "train_forward_passes_est": "1 student + MC teacher samples",
    },
    "snd": {
        "train_extra_networks": "history teacher + candidate/mix memory",
        "train_forward_passes_est": "2+",
    },
    "remask": {
        "train_extra_networks": "ema teacher + reliability/contrastive memories",
        "train_forward_passes_est": "2+ optional consistency branch",
    },
    "remask_dtst": {
        "train_extra_networks": "ema teacher + reliability/contrastive memories + DTST artifacts",
        "train_forward_passes_est": "2+ optional consistency branch",
    },
}

BASELINE_ARG_DEFAULTS = {
    "use_threshold": True,
    "threshold_beta": 0.001,
    "ema": None,
    "ema_update_period": None,
    "max_grad_norm": 10.0,
    "temperature": 1.0,
    "phase1_ratio": 0.25,
    "tt_stage2_lr_scale": 0.2,
    "tt_consistency_weight": 0.1,
    "entropy_weight": 0.01,
    "diversity_weight": 0.01,
    "consistency_weight": 1.0,
    "cross_weight": 0.5,
    "cal_top_ratio": 0.15,
    "cal_sce_a": 0.1,
    "cal_sce_b": 1.0,
    "cal_negative_width": 3,
    "ece_weight": 0.1,
    "ece_bins": 10,
    "class_weight_min": 0.5,
    "class_weight_max": 2.0,
    "prototype_weight": 0.1,
    "prototype_momentum": 0.99,
    "prototype_temperature": 0.1,
    "prototype_conf_threshold": 0.6,
    "prototype_max_pixels": 8192,
    "position_beta": 0.5,
    "crots_conf_threshold": 0.968,
    "sfmim_mi_weight": 0.1,
    "sfmim_conf_threshold": 0.70,
    "sfmim_class_weight_min": 0.25,
    "sfmim_class_weight_max": 4.0,
    "sfmim_patch_mask_ratio": 0.5,
    "sfmim_channel_mask_ratio": 0.5,
    "sfmim_patch_size": 64,
    "sfmim_scales": "1.0",
    "sfmim_whole_weight": 1.0,
    "sfmim_patch_weight": 1.0,
    "sfmim_channel_weight": 1.0,
    "sfmim_source_weight": 1.0,
    "sfmim_lovasz_weight": 1.0,
    "sfmim_soft_ce_weight": 1.0,
    "sfmim_entropy_eta": 2.0,
    "uotsl_mc_samples": 5,
    "uotsl_threshold": 0.90,
    "uotsl_kl_weight": 0.0,
    "uotsl_uncertainty_power": 1.0,
}

BASELINE_USED_HPARAMS = {
    "cal_sfda": [
        "use_threshold",
        "threshold_beta",
        "ema",
        "ema_update_period",
        "entropy_weight",
        "cal_top_ratio",
        "cal_sce_a",
        "cal_sce_b",
        "cal_negative_width",
        "ece_weight",
        "ece_bins",
    ],
    "tt_sfuda": [
        "use_threshold",
        "threshold_beta",
        "ema",
        "ema_update_period",
        "phase1_ratio",
        "tt_stage2_lr_scale",
        "tt_consistency_weight",
        "entropy_weight",
        "diversity_weight",
    ],
    "crossmatch": [
        "use_threshold",
        "threshold_beta",
        "ema",
        "ema_update_period",
        "temperature",
        "consistency_weight",
        "cross_weight",
        "max_grad_norm",
    ],
    "crots": [
        "use_threshold",
        "threshold_beta",
        "ema",
        "ema_update_period",
        "position_beta",
        "crots_conf_threshold",
    ],
    "iapc": [
        "use_threshold",
        "threshold_beta",
        "ema",
        "ema_update_period",
        "prototype_weight",
        "prototype_momentum",
        "prototype_temperature",
        "prototype_conf_threshold",
        "prototype_max_pixels",
        "entropy_weight",
    ],
    "sfmim": [
        "ema",
        "ema_update_period",
        "entropy_weight",
        "sfmim_conf_threshold",
        "sfmim_class_weight_min",
        "sfmim_class_weight_max",
        "sfmim_patch_mask_ratio",
        "sfmim_channel_mask_ratio",
        "sfmim_patch_size",
        "sfmim_scales",
        "sfmim_whole_weight",
        "sfmim_patch_weight",
        "sfmim_channel_weight",
        "sfmim_source_weight",
        "sfmim_lovasz_weight",
        "sfmim_soft_ce_weight",
        "sfmim_entropy_eta",
        "sfmim_mi_weight",
    ],
    "uotsl": [
        "use_threshold",
        "threshold_beta",
        "ema",
        "ema_update_period",
        "uotsl_mc_samples",
        "uotsl_threshold",
        "uotsl_kl_weight",
        "uotsl_uncertainty_power",
    ],
}

COMMON_CFG_HPARAMS = [
    "SOLVER.BASE_LR",
    "SOLVER.LR_METHOD",
    "SOLVER.LR_POWER",
    "SOLVER.MOMENTUM",
    "SOLVER.WEIGHT_DECAY",
    "SOLVER.MAX_ITER",
    "SOLVER.STOP_ITER",
    "SOLVER.BATCH_SIZE",
    "SOLVER.CHECKPOINT_PERIOD",
    "INPUT.SOURCE_INPUT_SIZE_TRAIN",
    "INPUT.TARGET_INPUT_SIZE_TRAIN",
    "INPUT.INPUT_SCALES_TRAIN",
    "INPUT.HORIZONTAL_FLIP_PROB_TRAIN",
    "INPUT.BRIGHTNESS",
    "INPUT.CONTRAST",
    "INPUT.SATURATION",
    "INPUT.HUE",
    "DTU.EMA_WEIGHT",
    "DTU.FIX_ITERATION",
    "DTU.DYNAMIC",
    "DTU.PROXY_METRIC",
    "TCR.OPEN",
    "TCR.TOPK_CANDIDATE",
    "TCR.UPDATE_FREQUENCY",
    "METAPL.OPEN",
    "METAPL.VAL_UPDATE",
]

REMASK_CFG_HPARAMS = [
    "REMASK.USE_RELIABILITY",
    "REMASK.RELIABILITY_MODE",
    "REMASK.ALPHA",
    "REMASK.BETA",
    "REMASK.TAU",
    "REMASK.DELTA",
    "REMASK.KL_SCALE",
    "REMASK.KL_SIGMOID",
    "REMASK.NORM_MOMENTUM",
    "REMASK.USE_CONTRASTIVE",
    "REMASK.LAMBDA",
    "REMASK.LAMBDA_N",
    "REMASK.MU",
    "REMASK.GAMMA",
    "REMASK.NEIGHBOR_TOPK",
    "REMASK.HISTORY_MOMENTUM",
    "REMASK.HISTORY_QUEUE_SIZE",
    "REMASK.HISTORY_UPDATE_PERIOD",
    "REMASK.HISTORY_UPDATE_MODE",
    "REMASK.HISTORY_STRIDE",
    "REMASK.HISTORY_USE_CONF",
    "REMASK.TEMPORAL_FLOOR",
    "REMASK.TEMPORAL_CONF_POWER",
    "REMASK.LOSS_WEIGHT_FLOOR",
    "REMASK.USE_ANCHOR",
    "REMASK.ANCHOR_FLOOR",
    "REMASK.ANCHOR_MODE",
    "REMASK.USE_CLASS_BALANCE",
    "REMASK.CLASS_BALANCE_GAMMA",
    "REMASK.CLASS_BALANCE_MIN",
    "REMASK.CLASS_BALANCE_MAX",
    "REMASK.USE_CLASS_CALIBRATION",
    "REMASK.CLASS_CALIBRATION_MOMENTUM",
    "REMASK.CLASS_CALIBRATION_GAMMA",
    "REMASK.CLASS_CALIBRATION_MIN",
    "REMASK.CLASS_CALIBRATION_MAX",
    "REMASK.USE_RELIABILITY_LOGIT_ADJUST",
    "REMASK.RELIABILITY_ADJUST_LAMBDA",
    "REMASK.RELIABILITY_ADJUST_PRIOR_FLOOR",
    "REMASK.UNDERWATER_PHYSICS_AUG",
    "REMASK.UNDERWATER_ATTENUATION",
    "REMASK.UNDERWATER_HAZE",
    "REMASK.UNDERWATER_TURBIDITY",
    "REMASK.UNDERWATER_CLASS_FOCUS",
    "REMASK.UNDERWATER_FOCUS_CLASSES",
    "REMASK.UNDERWATER_CLASS_FOCUS_MAX",
    "REMASK.UNDERWATER_CLASS_FOCUS_CONF",
]

REMASK_UNDERWATER_SCRIPT_HPARAMS = {
    "underwater_preset": "baseline",
    "underwater_factor_consistency": True,
    "underwater_factor_warmup_ratio": 0.35,
    "underwater_factor_lambda": 0.02,
    "underwater_app_weight": 0.5,
    "underwater_vis_weight": 1.0,
    "underwater_mix_weight": 0.75,
    "underwater_app_max_strength": 0.35,
    "underwater_vis_max_strength": 0.45,
    "underwater_revision_enabled": False,
    "underwater_revision_warmup_ratio": 0.20,
    "underwater_revision_alpha": 0.55,
    "underwater_revision_margin": 0.12,
    "underwater_revision_confidence": 0.55,
    "underwater_revision_local_kernel": 7,
    "underwater_revision_local_support": 0.60,
    "underwater_revision_aug_tolerance": 0.02,
    "underwater_revision_aug_confidence": 0.35,
    "underwater_revision_max_ratio": 0.08,
    "underwater_soft_conflict_enabled": False,
    "underwater_soft_conflict_margin": 0.04,
    "underwater_soft_conflict_weight": 0.25,
    "underwater_revision_use_aug_gate": True,
    "underwater_revision_use_boundary_gate": True,
    "underwater_revision_use_cap": True,
}

CLASS_NAMES = {
    0: "Background",
    1: "Human divers",
    2: "Aquatic plants/sea-grass",
    3: "Wrecks/ruins",
    4: "Robots/instruments",
    5: "Reefs and invertebrates",
    6: "Fish and vertebrates",
    7: "Sea-floor and rocks",
}

SUIM_PALETTE = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 0),
    3: (0, 255, 255),
    4: (255, 0, 0),
    5: (255, 0, 255),
    6: (255, 255, 0),
    7: (255, 255, 255),
    255: (128, 128, 128),
}

VAL_RE = re.compile(
    r"Val result(?: iter (?P<iter>\d+))?: mIoU/mAcc/allAcc "
    r"(?P<miou>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)/"
    r"(?P<macc>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)/"
    r"(?P<allacc>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)

EARLY_STOP_VAL_RE = re.compile(
    r"Early-stop monitor:.*?\bmIoU\s+"
    r"(?P<miou>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+at iter\s+"
    r"(?P<iter>\d+)"
)


def parse_size(text: str) -> Tuple[int, int]:
    cleaned = str(text).lower().replace(",", "x").replace(" ", "")
    if "x" not in cleaned:
        raise argparse.ArgumentTypeError("Expected size formatted as WIDTHxHEIGHT, e.g. 960x480")
    width, height = cleaned.split("x", 1)
    return int(width), int(height)


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_as_jsonable)


def acquire_lock(path: Path, timeout_s: float = 120.0) -> Path:
    lock = path.with_name(path.name + ".lock")
    deadline = time.time() + timeout_s
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return lock
        except FileExistsError:
            if time.time() >= deadline:
                raise TimeoutError("Timed out waiting for CSV lock: {}".format(lock))
            time.sleep(0.2)


def append_csv_row(path: Path, row: Dict[str, Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(path)
    try:
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def method_family(method: str) -> str:
    if method in BASELINE_METHODS:
        return "baseline"
    if method == "snd":
        return "snd"
    if method in {"remask", "remask_dtst"}:
        return "remask"
    raise ValueError("Unsupported method: {}".format(method))


def build_cfg(args: argparse.Namespace, output_dir: Optional[Path] = None, max_iter: Optional[int] = None) -> Any:
    cfg = DEFAULT_CFG.clone()
    cfg.merge_from_file(args.config_file)

    if args.method in BASELINE_METHODS and args.paper_preset:
        from train_sfda_baselines import _apply_paper_preset_cfg

        _apply_paper_preset_cfg(cfg, args.method)

    opts = [
        "DATASETS.SOURCE_TRAIN",
        args.source_train,
        "DATASETS.TARGET_TRAIN",
        args.target_train,
        "DATASETS.TEST",
        args.test_set,
        "DATASETS.TT",
        args.tt_set,
        "DATASETS.II",
        args.ii_set,
        "resume",
        args.source_checkpoint,
        "SOLVER.BATCH_SIZE",
        str(args.batch_size),
    ]
    if output_dir is not None:
        opts.extend(["OUTPUT_DIR", str(output_dir)])
    if max_iter is not None:
        opts.extend(
            [
                "SOLVER.MAX_ITER",
                str(max_iter),
                "SOLVER.STOP_ITER",
                str(args.profile_batches),
                "SOLVER.CHECKPOINT_PERIOD",
                str(max(args.profile_batches + 1000, 999999)),
            ]
        )
    if args.opts:
        opts.extend(args.opts)
    cfg.merge_from_list(opts)
    apply_underwater_label_args(cfg, args)
    cfg.freeze()
    return cfg


def _json_compact(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_as_jsonable)


def _cfg_get_path(cfg: Any, path: str, default: Any = "") -> Any:
    node = cfg
    for item in path.split("."):
        try:
            node = getattr(node, item)
        except Exception:
            return default
    return node


def _cfg_snapshot(cfg: Any, paths: Sequence[str]) -> Dict[str, Any]:
    return {path: _cfg_get_path(cfg, path) for path in paths if _cfg_get_path(cfg, path, None) is not None}


def _opts_has_key(opts: Optional[Sequence[str]], key: str) -> bool:
    return key in set(opts or [])


def _short_profile_max_iter(args: argparse.Namespace) -> int:
    family = method_family(args.method)
    return args.profile_batches + 1 if family in {"snd", "remask"} else args.profile_batches


def _effective_cfg(
    args: argparse.Namespace,
    out_dir: Path,
    max_iter: Optional[int] = None,
    profile_short_train: bool = False,
) -> Any:
    cfg = build_cfg(args, output_dir=out_dir, max_iter=max_iter)
    try:
        cfg.defrost()
    except Exception:
        pass

    family = method_family(args.method)
    if family == "baseline":
        cfg.TCR.OPEN = False
        cfg.METAPL.OPEN = False
    elif family == "snd" and profile_short_train and not args.profile_snd_metapl:
        cfg.METAPL.OPEN = False
    elif family == "remask" and Path(args.remask_train_script).name == "train_ReMask_underwater.py":
        if not _opts_has_key(args.opts, "REMASK.UNDERWATER_PHYSICS_AUG"):
            cfg.REMASK.UNDERWATER_PHYSICS_AUG = True
        if not _opts_has_key(args.opts, "REMASK.UNDERWATER_CLASS_FOCUS"):
            cfg.REMASK.UNDERWATER_CLASS_FOCUS = False

    try:
        cfg.freeze()
    except Exception:
        pass
    return cfg


def _baseline_runtime_args(method: str, cfg: Any, paper_preset: bool) -> Tuple[Dict[str, Any], str]:
    runtime = dict(BASELINE_ARG_DEFAULTS)
    preset_source = ""
    if paper_preset:
        try:
            from train_sfda_baselines import PAPER_PRESETS

            preset = PAPER_PRESETS.get(method) or {}
            runtime.update(preset.get("args", {}))
            preset_source = preset.get("source", "")
        except Exception as exc:
            preset_source = "could_not_import_preset: {}".format(type(exc).__name__)
    if runtime.get("ema") is None:
        runtime["ema"] = _cfg_get_path(cfg, "DTU.EMA_WEIGHT")
    if runtime.get("ema_update_period") is None:
        runtime["ema_update_period"] = _cfg_get_path(cfg, "DTU.FIX_ITERATION")
    selected = BASELINE_USED_HPARAMS.get(method, sorted(runtime.keys()))
    return {key: runtime.get(key, "") for key in selected}, preset_source


def collect_hyperparameters(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    full_cfg = _effective_cfg(args, out_dir)
    profile_cfg = _effective_cfg(
        args,
        out_dir / "short_train",
        max_iter=_short_profile_max_iter(args),
        profile_short_train=True,
    )
    family = method_family(args.method)

    train_script = {
        "baseline": args.baseline_train_script,
        "snd": args.snd_train_script,
        "remask": args.remask_train_script,
    }[family]
    paper_batch_size = ""
    paper_preset_source = ""
    method_hparams: Dict[str, Any] = {}
    script_hparams: Dict[str, Any] = {}

    if family == "baseline":
        method_hparams, paper_preset_source = _baseline_runtime_args(args.method, full_cfg, args.paper_preset)
        if args.paper_preset:
            try:
                from train_sfda_baselines import PAPER_PRESETS

                paper_cfg = PAPER_PRESETS.get(args.method, {}).get("cfg", {})
                paper_batch_size = paper_cfg.get("SOLVER.BATCH_SIZE", "")
            except Exception:
                paper_batch_size = ""
    elif family == "snd":
        method_hparams = _cfg_snapshot(
            full_cfg,
            [
                "DTU.DYNAMIC",
                "DTU.Query_STEP",
                "DTU.Query_START",
                "DTU.META_MAX_UPDATE",
                "DTU.EMA_WEIGHT",
                "DTU.PROXY_METRIC",
                "DTU.FIX_ITERATION",
                "TCR.OPEN",
                "TCR.TOPK_CANDIDATE",
                "TCR.UPDATE_FREQUENCY",
                "METAPL.OPEN",
                "METAPL.VAL_UPDATE",
                "METAPL.TOP_K",
            ],
        )
        if profile_cfg.METAPL.OPEN != full_cfg.METAPL.OPEN:
            method_hparams["profile_short_train_override.METAPL.OPEN"] = profile_cfg.METAPL.OPEN
    else:
        method_hparams = _cfg_snapshot(full_cfg, REMASK_CFG_HPARAMS)
        if Path(args.remask_train_script).name == "train_ReMask_underwater.py":
            script_hparams = dict(REMASK_UNDERWATER_SCRIPT_HPARAMS)

    cfg_keys = list(COMMON_CFG_HPARAMS)
    if family == "remask":
        cfg_keys.extend(REMASK_CFG_HPARAMS)
    config_hparams = _cfg_snapshot(full_cfg, cfg_keys)

    base_lr = _cfg_get_path(full_cfg, "SOLVER.BASE_LR")
    classifier_lr_multiplier = 10.0
    row = {
        "method": args.method,
        "method_label": METHOD_LABELS.get(args.method, args.method),
        "method_family": family,
        "train_script": train_script,
        "config_file": args.config_file,
        "source_checkpoint": args.source_checkpoint,
        "source_train": args.source_train,
        "target_train": args.target_train,
        "test_set": args.test_set,
        "tt_set": args.tt_set,
        "ii_set": args.ii_set,
        "input_size": "{}x{}".format(args.input_size[0], args.input_size[1]),
        "profile_batches": int(args.profile_batches),
        "profile_batch_size": _cfg_get_path(profile_cfg, "SOLVER.BATCH_SIZE"),
        "profile_solver_max_iter": _cfg_get_path(profile_cfg, "SOLVER.MAX_ITER"),
        "profile_solver_stop_iter": _cfg_get_path(profile_cfg, "SOLVER.STOP_ITER"),
        "paper_preset": bool(args.paper_preset) if family == "baseline" else "",
        "paper_preset_source": paper_preset_source,
        "paper_batch_size": paper_batch_size,
        "optimizer": "SGD",
        "base_lr": base_lr,
        "classifier_lr": base_lr * classifier_lr_multiplier if isinstance(base_lr, (int, float)) else "",
        "classifier_lr_multiplier": classifier_lr_multiplier,
        "lr_method": _cfg_get_path(full_cfg, "SOLVER.LR_METHOD"),
        "lr_power": _cfg_get_path(full_cfg, "SOLVER.LR_POWER"),
        "momentum": _cfg_get_path(full_cfg, "SOLVER.MOMENTUM"),
        "weight_decay": _cfg_get_path(full_cfg, "SOLVER.WEIGHT_DECAY"),
        "solver_max_iter": _cfg_get_path(full_cfg, "SOLVER.MAX_ITER"),
        "solver_stop_iter": _cfg_get_path(full_cfg, "SOLVER.STOP_ITER"),
        "checkpoint_period": _cfg_get_path(full_cfg, "SOLVER.CHECKPOINT_PERIOD"),
        "ema": method_hparams.get("ema", _cfg_get_path(full_cfg, "DTU.EMA_WEIGHT")),
        "ema_update_period": method_hparams.get("ema_update_period", _cfg_get_path(full_cfg, "DTU.FIX_ITERATION")),
        "tcr_open": _cfg_get_path(full_cfg, "TCR.OPEN"),
        "tcr_update_frequency": _cfg_get_path(full_cfg, "TCR.UPDATE_FREQUENCY"),
        "metapl_open": _cfg_get_path(full_cfg, "METAPL.OPEN"),
        "metapl_open_profile": _cfg_get_path(profile_cfg, "METAPL.OPEN"),
        "profile_snd_metapl": bool(args.profile_snd_metapl) if family == "snd" else "",
        "dtu_dynamic": _cfg_get_path(full_cfg, "DTU.DYNAMIC"),
        "dtu_proxy_metric": _cfg_get_path(full_cfg, "DTU.PROXY_METRIC"),
        "dtu_fix_iteration": _cfg_get_path(full_cfg, "DTU.FIX_ITERATION"),
        "pl_threshold": args.pl_threshold,
        "analysis_max_batches": args.analysis_max_batches,
        "mask_samples": args.mask_samples,
        "train_extra_networks": TRAIN_COMPLEXITY_META.get(args.method, {}).get("train_extra_networks", ""),
        "train_forward_passes_est": TRAIN_COMPLEXITY_META.get(args.method, {}).get("train_forward_passes_est", ""),
        "method_hyperparams": _json_compact(method_hparams),
        "config_hyperparams": _json_compact(config_hparams),
        "script_hyperparams": _json_compact(script_hparams) if script_hparams else "",
        "opts": " ".join(args.opts or []),
    }
    return row


def load_checkpoint_into(cfg: Any, feature_extractor: torch.nn.Module, classifier: torch.nn.Module, path: str) -> None:
    if not path:
        return
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        return
    load_kwargs = {"map_location": torch.device("cpu")}
    try:
        # ReMask checkpoints are produced by this repository and include
        # non-tensor training state (for example collections.deque). PyTorch
        # 2.6 defaults to weights_only=True, which rejects that trusted state.
        checkpoint = torch.load(str(ckpt_path), weights_only=False, **load_kwargs)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        checkpoint = torch.load(str(ckpt_path), **load_kwargs)
    if "feature_extractor" not in checkpoint or "classifier" not in checkpoint:
        return
    feature_weights = strip_prefix_if_present(checkpoint["feature_extractor"], "module.")
    classifier_weights = strip_prefix_if_present(checkpoint["classifier"], "module.")
    feature_extractor.load_state_dict(feature_weights, strict=False)
    classifier.load_state_dict(classifier_weights, strict=False)


def count_parameters(modules: Sequence[torch.nn.Module]) -> Tuple[int, int, int]:
    total = 0
    trainable = 0
    bytes_fp32 = 0
    for module in modules:
        for param in module.parameters():
            count = param.numel()
            total += count
            if param.requires_grad:
                trainable += count
            bytes_fp32 += count * 4
    return total, trainable, bytes_fp32


class SegmentationWrapper(torch.nn.Module):
    def __init__(self, feature_extractor: torch.nn.Module, classifier: torch.nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _, features = self.feature_extractor(images)
        return self.classifier(features, images.shape[-2:])


def try_profile_ops(model: torch.nn.Module, dummy: torch.Tensor) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ops_status": "not_available",
        "ops_tool": "",
        "ops_unit": "",
        "ops": "",
        "ops_g": "",
        "ops_m": "",
    }
    try:
        from fvcore.nn import FlopCountAnalysis

        with torch.no_grad():
            flops = float(FlopCountAnalysis(model, dummy).total())
        result.update(
            {
                "ops_status": "ok",
                "ops_tool": "fvcore",
                "ops_unit": "FLOPs",
                "ops": flops,
                "ops_g": flops / 1e9,
                "ops_m": flops / 1e6,
            }
        )
        return result
    except Exception as exc:
        result["ops_status"] = "fvcore_failed: {}".format(type(exc).__name__)

    try:
        from thop import profile

        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        macs = float(macs)
        result.update(
            {
                "ops_status": "ok",
                "ops_tool": "thop",
                "ops_unit": "MACs",
                "ops": macs,
                "ops_g": macs / 1e9,
                "ops_m": macs / 1e6,
            }
        )
    except Exception as exc:
        result["ops_status"] = "{}; thop_failed: {}".format(result["ops_status"], type(exc).__name__)
    return result


def profile_static_and_inference(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    device = torch.device(args.device)
    cfg = build_cfg(args, output_dir=out_dir)
    feature_extractor = build_feature_extractor(cfg)
    classifier = build_classifier(cfg)
    total_params, trainable_params, bytes_fp32 = count_parameters([feature_extractor, classifier])

    feature_extractor.to(device)
    classifier.to(device)
    load_checkpoint_into(cfg, feature_extractor, classifier, args.source_checkpoint)
    feature_extractor.eval()
    classifier.eval()
    wrapper = SegmentationWrapper(feature_extractor, classifier).to(device).eval()

    width, height = args.input_size
    dummy = torch.randn(args.inference_batch_size, 3, height, width, device=device)

    ops = {}
    if args.measure_ops:
        ops = try_profile_ops(wrapper, dummy)

    latency_ms: List[float] = []
    peak_infer_mb = ""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(args.latency_warmup):
                _ = wrapper(dummy)
            torch.cuda.synchronize()
            for _ in range(args.latency_repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = wrapper(dummy)
                end.record()
                torch.cuda.synchronize()
                latency_ms.append(float(start.elapsed_time(end)))
        peak_infer_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
    else:
        with torch.no_grad():
            for _ in range(args.latency_warmup):
                _ = wrapper(dummy)
            for _ in range(args.latency_repeats):
                start_t = time.perf_counter()
                _ = wrapper(dummy)
                latency_ms.append((time.perf_counter() - start_t) * 1000.0)

    mean_ms = float(np.mean(latency_ms)) if latency_ms else math.nan
    std_ms = float(np.std(latency_ms)) if latency_ms else math.nan
    result = {
        "method": args.method,
        "method_label": METHOD_LABELS.get(args.method, args.method),
        "backbone": cfg.MODEL.NAME,
        "num_classes": int(cfg.MODEL.NUM_CLASSES),
        "input_width": width,
        "input_height": height,
        "inference_batch_size": int(args.inference_batch_size),
        "params": int(total_params),
        "params_m": total_params / 1e6,
        "trainable_params": int(trainable_params),
        "trainable_params_m": trainable_params / 1e6,
        "model_size_fp32_mb": bytes_fp32 / 1e6,
        "latency_ms_mean": mean_ms,
        "latency_ms_std": std_ms,
        "fps": 1000.0 / mean_ms if mean_ms and mean_ms > 0 else "",
        "peak_inference_memory_mb": peak_infer_mb,
        "extra_module_at_test": "no",
        "forward_passes_at_test": 1,
    }
    result.update(TRAIN_COMPLEXITY_META.get(args.method, {}))
    result.update(ops)
    del wrapper, feature_extractor, classifier, dummy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


class PowerSampler:
    def __init__(self, interval_s: float = 0.25, gpu_index: int = 0):
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self.samples: List[Tuple[float, Optional[float], Optional[float]]] = []
        self.status = "not_started"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml = None
        self._handle = None

    def _init_nvml(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(int(self.gpu_index))
            self.status = "pynvml"
            return True
        except Exception:
            self._nvml = None
            self._handle = None
            return False

    def _sample_nvml(self) -> Tuple[Optional[float], Optional[float]]:
        if self._nvml is None or self._handle is None:
            return None, None
        power_w = None
        mem_mb = None
        try:
            power_w = float(self._nvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
        except Exception:
            pass
        try:
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            mem_mb = float(mem.used) / 1024.0 / 1024.0
        except Exception:
            pass
        return power_w, mem_mb

    def _sample_nvidia_smi(self) -> Tuple[Optional[float], Optional[float]]:
        cmd = [
            "nvidia-smi",
            "-i",
            str(self.gpu_index),
            "--query-gpu=power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=5)
            first = out.strip().splitlines()[0]
            power_s, mem_s = [part.strip() for part in first.split(",", 1)]
            return float(power_s), float(mem_s)
        except Exception:
            return None, None

    def _loop(self) -> None:
        if not self._init_nvml():
            self.status = "nvidia-smi"
        while not self._stop.is_set():
            if self._nvml is not None:
                power_w, mem_mb = self._sample_nvml()
            else:
                power_w, mem_mb = self._sample_nvidia_smi()
                if power_w is None and mem_mb is None:
                    self.status = "unavailable"
            self.samples.append((time.time(), power_w, mem_mb))
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4))
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        energy_j = 0.0
        valid_power = []
        peak_mem = None
        previous_t = None
        previous_p = None
        for t, power_w, mem_mb in self.samples:
            if mem_mb is not None:
                peak_mem = mem_mb if peak_mem is None else max(peak_mem, mem_mb)
            if power_w is None:
                previous_t = t
                previous_p = None
                continue
            valid_power.append(power_w)
            if previous_t is not None and previous_p is not None:
                energy_j += (t - previous_t) * (previous_p + power_w) * 0.5
            previous_t = t
            previous_p = power_w
        return {
            "power_sampler": self.status,
            "power_samples": len(self.samples),
            "train_energy_j": energy_j if valid_power else "",
            "train_energy_wh": energy_j / 3600.0 if valid_power else "",
            "train_power_w_mean": float(np.mean(valid_power)) if valid_power else "",
            "train_power_w_peak": float(np.max(valid_power)) if valid_power else "",
            "train_peak_gpu_memory_mb_sampled": peak_mem if peak_mem is not None else "",
        }


def has_warm_artifacts(path: Path) -> bool:
    return (
        (path / "CTR.p").is_file()
        and (path / "CTR_O.p").is_file()
        and (path / "CTR").is_dir()
        and (path / "CTR_O").is_dir()
    )


def has_snd_meta_artifacts(path: Path) -> bool:
    meta_dir = path / "meta_val"
    return meta_dir.is_dir() and any(meta_dir.glob("*.png"))


def copy_warm_artifacts(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("CTR.p", "CTR_O.p"):
        shutil.copy2(source / filename, target / filename)
    for dirname in ("CTR", "CTR_O"):
        src_dir = source / dirname
        dst_dir = target / dirname
        if dst_dir.exists():
            continue
        shutil.copytree(src_dir, dst_dir)
    for dirname in ("meta_val", "meta_val_mixup"):
        src_dir = source / dirname
        dst_dir = target / dirname
        if src_dir.is_dir() and not dst_dir.exists():
            shutil.copytree(src_dir, dst_dir)
    for filename in ("meta_val.p", "meta_val_mixup.p"):
        src_file = source / filename
        if src_file.is_file():
            shutil.copy2(src_file, target / filename)


def _write_label_info(label_dir: Path, out_path: Path, num_classes: int) -> int:
    label_to_file = {idx: [] for idx in range(num_classes)}
    file_to_label: Dict[str, List[int]] = {}
    for item in sorted(label_dir.glob("*.png")):
        try:
            label = np.asarray(Image.open(item))
        except Exception:
            continue
        classes = sorted(int(value) for value in np.unique(label) if 0 <= int(value) < num_classes)
        if not classes:
            continue
        file_to_label[item.name] = classes
        for class_id in classes:
            label_to_file[class_id].append(item.name)
    with out_path.open("wb") as handle:
        pickle.dump((label_to_file, file_to_label), handle)
    return len(file_to_label)


def synthesize_snd_meta_artifacts(args: argparse.Namespace, train_dir: Path) -> None:
    source = train_dir / "CTR"
    if not source.is_dir():
        source = train_dir / "CTR_O"
    if not source.is_dir():
        return

    try:
        cfg_for_meta = build_cfg(args, output_dir=train_dir)
        num_classes = int(_cfg_get_path(cfg_for_meta, "MODEL.NUM_CLASSES", 8))
        top_k = float(_cfg_get_path(cfg_for_meta, "METAPL.TOP_K", 0.01))
    except Exception:
        num_classes = 8
        top_k = 0.01

    pngs = sorted(source.glob("*.png"))
    if not pngs:
        return
    max_items = max(1, int(math.ceil(len(pngs) * max(top_k, 0.01))))
    selected = pngs[:max_items]
    for dirname in ("meta_val", "meta_val_mixup"):
        dst_dir = train_dir / dirname
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not any(dst_dir.glob("*.png")):
            for item in selected:
                shutil.copy2(item, dst_dir / item.name)
        _write_label_info(dst_dir, train_dir / "{}.p".format(dirname), num_classes)
    (train_dir / "profile_snd_meta_val_synthesized.txt").write_text(
        "Synthetic meta_val/meta_val_mixup were copied from {} for short SND METAPL profiling only.\n".format(
            source
        ),
        encoding="utf-8",
    )


def ensure_snd_meta_artifacts(args: argparse.Namespace, train_dir: Path) -> None:
    if method_family(args.method) != "snd" or not args.profile_snd_metapl:
        return
    if has_snd_meta_artifacts(train_dir):
        return
    if args.warm_artifact_dir:
        warm = Path(args.warm_artifact_dir)
        if has_snd_meta_artifacts(warm):
            copy_warm_artifacts(warm, train_dir)
            return
    synthesize_snd_meta_artifacts(args, train_dir)


def prepare_training_dir(args: argparse.Namespace, train_dir: Path) -> None:
    train_dir.mkdir(parents=True, exist_ok=True)
    if method_family(args.method) not in {"snd", "remask"}:
        return
    if not has_warm_artifacts(train_dir) and args.warm_artifact_dir:
        warm = Path(args.warm_artifact_dir)
        if has_warm_artifacts(warm):
            copy_warm_artifacts(warm, train_dir)
    if not has_warm_artifacts(train_dir) and not args.allow_candidate_build:
        raise RuntimeError(
            "SND/ReMask profiling needs CTR/CTR_O warm artifacts. "
            "Set --warm-artifact-dir or pass --allow-candidate-build to build them during profiling."
        )
    ensure_snd_meta_artifacts(args, train_dir)


def build_training_command(args: argparse.Namespace, train_dir: Path) -> List[str]:
    family = method_family(args.method)
    max_iter = args.profile_batches + 1 if family in {"snd", "remask"} else args.profile_batches
    common_opts = [
        "-cfg",
        args.config_file,
        "resume",
        args.source_checkpoint,
        "DATASETS.SOURCE_TRAIN",
        args.source_train,
        "DATASETS.TARGET_TRAIN",
        args.target_train,
        "DATASETS.TEST",
        args.test_set,
        "DATASETS.TT",
        args.tt_set,
        "DATASETS.II",
        args.ii_set,
        "OUTPUT_DIR",
        str(train_dir),
        "SOLVER.BATCH_SIZE",
        str(args.batch_size),
        "SOLVER.MAX_ITER",
        str(max_iter),
        "SOLVER.STOP_ITER",
        str(args.profile_batches),
        "SOLVER.CHECKPOINT_PERIOD",
        str(max(args.profile_batches + 1000, 999999)),
    ]
    if args.opts:
        common_opts.extend(args.opts)

    if family == "baseline":
        cmd = [
            args.python,
            args.baseline_train_script,
            "--skip-test",
            "--no-checkpoint",
            "--method",
            args.method,
        ]
        cmd.append("--paper-preset" if args.paper_preset else "--no-paper-preset")
        cmd.extend(common_opts)
        return cmd
    if family == "snd":
        if not args.profile_snd_metapl:
            common_opts.extend(["METAPL.OPEN", "False"])
        return [args.python, args.snd_train_script, "--skip-test"] + common_opts
    return [args.python, args.remask_train_script, "--skip-test"] + common_opts


def parse_training_log(text: str) -> Dict[str, Any]:
    max_mem = ""
    matches = re.findall(r"max mem:\s*([0-9.]+)", text)
    if matches:
        max_mem = max(float(item) for item in matches)
    s_per_it = ""
    total_match = re.search(r"Total .*training time: .*?\(([0-9.]+)\s+s\s*/\s+it\)", text)
    if total_match:
        s_per_it = float(total_match.group(1))
    return {
        "train_peak_cuda_allocated_mb_logged": max_mem,
        "train_s_per_iter_logged": s_per_it,
    }


def run_training_profile(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    if args.skip_training_profile:
        return {"training_profile_status": "skipped"}

    train_dir = out_dir / "short_train"
    prepare_training_dir(args, train_dir)
    cmd = build_training_command(args, train_dir)
    stdout_path = out_dir / "train_profile_stdout.log"
    command_path = out_dir / "train_profile_command.txt"
    failed_path = out_dir / "training_failed.json"
    if failed_path.is_file():
        failed_path.unlink()
    command_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")

    env = os.environ.copy()
    repo_root = str(Path.cwd())
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    sampler = PowerSampler(interval_s=args.power_sample_interval, gpu_index=args.gpu_index)
    start = time.perf_counter()
    sampler.start()
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.perf_counter() - start
    power = sampler.stop()
    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")

    parsed = parse_training_log(proc.stdout)
    result: Dict[str, Any] = {
        "training_profile_status": "ok" if proc.returncode == 0 else "failed",
        "training_returncode": proc.returncode,
        "profile_batches": int(args.profile_batches),
        "synthetic_snd_meta_val": bool((train_dir / "profile_snd_meta_val_synthesized.txt").is_file()),
        "train_elapsed_s_wall": elapsed,
        "train_s_per_batch_wall": elapsed / max(1, args.profile_batches),
        "train_command": " ".join(cmd),
        "train_stdout": str(stdout_path),
    }
    result.update(power)
    result.update(parsed)
    if proc.returncode != 0:
        result["training_error_tail"] = "\n".join(proc.stdout.splitlines()[-60:])
        write_json(failed_path, result)
    return result


def find_checkpoint(run_dir: Optional[str], method: str) -> str:
    if not run_dir:
        return ""
    root = Path(run_dir)
    if root.is_file():
        return str(root)
    if not root.is_dir():
        return ""

    preferred = [
        "{}_model_best.pth".format(method),
        "remask_model_best.pth",
        "model_best.pth",
    ]
    for name in preferred:
        path = root / name
        if path.is_file():
            return str(path)

    patterns = [
        "*_model_best.pth",
        "remask_model_iter*.pth",
        "{}_model_iter*.pth".format(method),
        "model_iter*.pth",
        "*.pth",
    ]
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(root.glob(pattern)))
        if candidates:
            break
    if not candidates:
        return ""

    def iter_key(path: Path) -> Tuple[int, float]:
        match = re.search(r"iter(\d+)", path.name)
        iteration = int(match.group(1)) if match else -1
        return iteration, path.stat().st_mtime

    best_iter = _best_validation_iteration(root)
    if best_iter is not None:
        candidates_with_iter = [(path, iter_key(path)[0]) for path in candidates if iter_key(path)[0] >= 0]
        if candidates_with_iter:
            return str(min(candidates_with_iter, key=lambda item: (abs(item[1] - best_iter), -item[1]))[0])

    return str(max(candidates, key=iter_key))


def _validation_rows_from_log(log_path: Path) -> List[Dict[str, Any]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rows: List[Dict[str, Any]] = []
    last_iter: Optional[int] = None
    pending_row: Optional[int] = None

    for line in text.splitlines():
        iter_match = re.search(r"\biter:\s*(\d+)", line)
        if iter_match:
            current_iter = int(iter_match.group(1))
            # A Slurm log may contain sequential Stage-A and Stage-B runs. A
            # decreasing train iteration starts a new run, so retain only the
            # latest segment requested by --analysis-run-dir.
            if last_iter is not None and current_iter < last_iter:
                rows.clear()
                pending_row = None
            # Validation is emitted after the last periodically logged train
            # step. The next train step therefore pins the pending val to N-1.
            if pending_row is not None and current_iter > int(rows[pending_row]["iteration"] or -1):
                rows[pending_row]["iteration"] = str(current_iter - 1)
                pending_row = None
            last_iter = current_iter

        val = VAL_RE.search(line)
        if val:
            rows.append(
                {
                    "iteration": val.group("iter") or (str(last_iter) if last_iter is not None else ""),
                    "mIoU": val.group("miou"),
                    "mAcc": val.group("macc"),
                    "allAcc": val.group("allacc"),
                    "checkpoint": "",
                }
            )
            pending_row = None if val.group("iter") else len(rows) - 1
            continue

        monitor = EARLY_STOP_VAL_RE.search(line)
        if monitor and rows:
            candidate = pending_row if pending_row is not None else len(rows) - 1
            if abs(float(rows[candidate]["mIoU"]) - float(monitor.group("miou"))) < 5e-5:
                rows[candidate]["iteration"] = monitor.group("iter")
                pending_row = None

    deduplicated: List[Dict[str, Any]] = []
    for row in rows:
        if deduplicated and row["iteration"] == deduplicated[-1]["iteration"]:
            deduplicated[-1] = row
        else:
            deduplicated.append(row)
    return deduplicated


def _best_validation_iteration(run_dir: Path) -> Optional[int]:
    rows: List[Dict[str, Any]] = []
    metrics_csv = run_dir / "metrics.csv"
    if metrics_csv.is_file():
        with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    else:
        log_path = run_dir / "log.txt"
        if log_path.is_file():
            rows.extend(_validation_rows_from_log(log_path))
    parsed: List[Tuple[float, int]] = []
    for row in rows:
        try:
            miou = float(row.get("mIoU", ""))
            iteration = int(float(row.get("iteration", "")))
        except (TypeError, ValueError):
            continue
        parsed.append((miou, iteration))
    if not parsed:
        return None
    return max(parsed, key=lambda item: item[0])[1]


def validation_curve(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    if not args.analysis_run_dir:
        return {}
    run_dir = Path(args.analysis_run_dir)
    if not run_dir.is_dir():
        return {}

    rows: List[Dict[str, Any]] = []
    metrics_csv = run_dir / "metrics.csv"
    if metrics_csv.is_file():
        with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "iteration": row.get("iteration", ""),
                        "mIoU": row.get("mIoU", ""),
                        "mAcc": row.get("mAcc", ""),
                        "allAcc": row.get("allAcc", ""),
                        "checkpoint": row.get("checkpoint", ""),
                    }
                )
    else:
        log_path = run_dir / "log.txt"
        if log_path.is_file():
            rows.extend(_validation_rows_from_log(log_path))

    if not rows:
        return {}

    curve_path = out_dir / "validation_curve.csv"
    with curve_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "mIoU", "mAcc", "allAcc", "checkpoint"])
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda row: float(row["mIoU"]) if row.get("mIoU") not in ("", None) else -1)
    return {
        "validation_curve_csv": str(curve_path),
        "validation_best_iter": best.get("iteration", ""),
        "validation_best_miou": float(best["mIoU"]) if best.get("mIoU") not in ("", None) else "",
    }


def _safe_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else math.nan


def _class_groups(freq: np.ndarray) -> Dict[str, List[int]]:
    foreground = [idx for idx in range(1, len(freq)) if freq[idx] > 0]
    if not foreground:
        return {"head": [], "medium": [], "tail": []}
    ranked = sorted(foreground, key=lambda idx: freq[idx])
    group_size = max(1, int(math.ceil(len(ranked) / 3.0)))
    tail = ranked[:group_size]
    head = ranked[-group_size:]
    medium = [idx for idx in ranked if idx not in set(tail) and idx not in set(head)]
    return {"head": head, "medium": medium, "tail": tail}


def _safe_stem(name: Any) -> str:
    if isinstance(name, (list, tuple)):
        name = name[0] if name else "sample"
    name = str(name)
    stem = Path(name).stem or "sample"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "sample"


def _label_to_rgb(label: np.ndarray) -> Image.Image:
    label = label.astype(np.int64)
    rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for class_id, color in SUIM_PALETTE.items():
        rgb[label == class_id] = color
    unknown = ~np.isin(label, list(SUIM_PALETTE.keys()))
    rgb[unknown] = SUIM_PALETTE[255]
    return Image.fromarray(rgb, mode="RGB")


def _colorize_map(values: np.ndarray, valid: np.ndarray, vmin: float, vmax: float) -> Image.Image:
    arr = values.astype(np.float32)
    valid = valid.astype(bool)
    denom = max(float(vmax - vmin), 1e-8)
    norm = np.clip((arr - float(vmin)) / denom, 0.0, 1.0)
    stops = np.asarray(
        [
            (49, 54, 149),
            (69, 117, 180),
            (116, 173, 209),
            (224, 243, 248),
            (255, 255, 191),
            (254, 224, 144),
            (244, 109, 67),
            (165, 0, 38),
        ],
        dtype=np.float32,
    )
    scaled = norm * (len(stops) - 1)
    lo = np.floor(scaled).astype(np.int64)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    alpha = (scaled - lo)[..., None]
    rgb = stops[lo] * (1.0 - alpha) + stops[hi] * alpha
    rgb = rgb.astype(np.uint8)
    rgb[~valid] = (128, 128, 128)
    return Image.fromarray(rgb, mode="RGB")


def _tensor_heatmap(
    values: torch.Tensor,
    valid: torch.Tensor,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Image.Image:
    return _colorize_map(
        values.detach().cpu().float().numpy(),
        valid.detach().cpu().bool().numpy(),
        vmin,
        vmax,
    )


def _tensor_error_map(pred: torch.Tensor, label: torch.Tensor, valid: torch.Tensor) -> Image.Image:
    pred_np = pred.detach().cpu().numpy()
    label_np = label.detach().cpu().numpy()
    valid_np = valid.detach().cpu().bool().numpy()
    correct = (pred_np == label_np) & valid_np
    wrong = (pred_np != label_np) & valid_np
    rgb = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
    rgb[~valid_np] = (128, 128, 128)
    rgb[correct] = (74, 155, 84)
    rgb[wrong] = (220, 55, 55)
    return Image.fromarray(rgb, mode="RGB")


def _pl_selection_map(
    selected: torch.Tensor,
    pred: torch.Tensor,
    label: torch.Tensor,
    valid: torch.Tensor,
) -> Image.Image:
    selected_np = selected.detach().cpu().bool().numpy()
    pred_np = pred.detach().cpu().numpy()
    label_np = label.detach().cpu().numpy()
    valid_np = valid.detach().cpu().bool().numpy()
    correct_selected = selected_np & valid_np & (pred_np == label_np)
    wrong_selected = selected_np & valid_np & (pred_np != label_np)
    unselected = (~selected_np) & valid_np
    rgb = np.zeros((*pred_np.shape, 3), dtype=np.uint8)
    rgb[~valid_np] = (128, 128, 128)
    rgb[unselected] = (38, 70, 120)
    rgb[correct_selected] = (72, 175, 92)
    rgb[wrong_selected] = (225, 72, 72)
    return Image.fromarray(rgb, mode="RGB")


def _image_to_pil(image: torch.Tensor, cfg: Any) -> Image.Image:
    image = image.detach().cpu().float().clone()
    mean = torch.tensor(cfg.INPUT.PIXEL_MEAN, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(cfg.INPUT.PIXEL_STD, dtype=image.dtype).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0.0, 1.0)
    arr = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _sample_miou(pred: torch.Tensor, label: torch.Tensor, num_classes: int, ignore_label: int) -> float:
    valid = label != ignore_label
    values = []
    for class_id in range(num_classes):
        pred_c = (pred == class_id) & valid
        label_c = (label == class_id) & valid
        union = (pred_c | label_c).sum().item()
        if union <= 0:
            continue
        inter = (pred_c & label_c).sum().item()
        values.append(float(inter) / float(union))
    return float(np.mean(values)) if values else math.nan


def _write_single_row_csv(path: Path, row: Dict[str, Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def analyze_checkpoint(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    if not args.extra_analysis:
        return {"extra_analysis_status": "skipped"}

    checkpoint = args.analysis_checkpoint or find_checkpoint(args.analysis_run_dir, args.method)
    if not checkpoint or not Path(checkpoint).is_file():
        if args.strict_extra_analysis:
            raise FileNotFoundError(
                "Extra analysis needs a checkpoint for {}. Set --analysis-run-dir or "
                "--analysis-checkpoint to a folder/file that exists on HPC.".format(args.method)
            )
        return {
            "extra_analysis_status": "skipped_no_checkpoint",
            "analysis_run_dir": args.analysis_run_dir or "",
            "analysis_checkpoint": checkpoint or "",
        }

    device = torch.device(args.device)
    cfg = build_cfg(args, output_dir=out_dir)
    feature_extractor = build_feature_extractor(cfg).to(device)
    classifier = build_classifier(cfg).to(device)
    load_checkpoint_into(cfg, feature_extractor, classifier, checkpoint)
    feature_extractor.eval()
    classifier.eval()

    dataset = build_dataset(cfg, mode="test", is_source=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.analysis_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    num_classes = int(cfg.MODEL.NUM_CLASSES)
    ignore = int(cfg.INPUT.IGNORE_LABEL)
    intersection = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    target = np.zeros(num_classes, dtype=np.float64)
    pl_intersection = np.zeros(num_classes, dtype=np.float64)
    pl_union = np.zeros(num_classes, dtype=np.float64)
    pl_target = np.zeros(num_classes, dtype=np.float64)
    freq = np.zeros(num_classes, dtype=np.float64)

    total_valid = 0.0
    selected_pixels = 0.0
    selected_correct = 0.0
    entropy_sum = 0.0
    conf_correct_sum = 0.0
    conf_wrong_sum = 0.0
    correct_count = 0.0
    wrong_count = 0.0
    ece_bins = np.zeros(args.ece_bins, dtype=np.float64)
    ece_conf = np.zeros(args.ece_bins, dtype=np.float64)
    ece_acc = np.zeros(args.ece_bins, dtype=np.float64)
    mask_dir = out_dir / "mask_inference"
    mask_rows: List[Dict[str, Any]] = []
    save_masks = (not args.no_mask_inference) and args.mask_samples > 0
    if save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, labels, names) in enumerate(loader):
            if args.analysis_max_batches > 0 and batch_idx >= args.analysis_max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            _, features = feature_extractor(images)
            logits = classifier(features, labels.shape[-2:])
            prob = F.softmax(logits, dim=1)
            conf, pred = prob.max(dim=1)
            valid = labels != ignore
            selected = valid & (conf >= args.pl_threshold)
            entropy = (-prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1)

            inter, uni, tgt = intersectionAndUnionGPU(pred.clone(), labels, num_classes, ignore)
            intersection += inter.detach().cpu().numpy()
            union += uni.detach().cpu().numpy()
            target += tgt.detach().cpu().numpy()

            selected_labels = labels.clone()
            selected_labels[~selected] = ignore
            pl_inter, pl_uni, pl_tgt = intersectionAndUnionGPU(pred.clone(), selected_labels, num_classes, ignore)
            pl_intersection += pl_inter.detach().cpu().numpy()
            pl_union += pl_uni.detach().cpu().numpy()
            pl_target += pl_tgt.detach().cpu().numpy()

            valid_count = float(valid.sum().item())
            total_valid += valid_count
            selected_pixels += float(selected.sum().item())
            selected_correct += float(((pred == labels) & selected).sum().item())
            entropy_sum += float(entropy[valid].sum().item())
            for cls_id in range(num_classes):
                freq[cls_id] += float(((labels == cls_id) & valid).sum().item())

            correctness = (pred == labels) & valid
            wrong = (~correctness) & valid
            correct_count += float(correctness.sum().item())
            wrong_count += float(wrong.sum().item())
            conf_correct_sum += float(conf[correctness].sum().item())
            conf_wrong_sum += float(conf[wrong].sum().item())

            flat_conf = conf[valid].detach()
            flat_acc = (pred[valid] == labels[valid]).float()
            if flat_conf.numel() > 0:
                bin_ids = torch.clamp((flat_conf * args.ece_bins).long(), max=args.ece_bins - 1)
                for bin_id in range(args.ece_bins):
                    mask = bin_ids == bin_id
                    count = float(mask.sum().item())
                    if count <= 0:
                        continue
                    ece_bins[bin_id] += count
                    ece_conf[bin_id] += float(flat_conf[mask].sum().item())
                    ece_acc[bin_id] += float(flat_acc[mask].sum().item())

            if save_masks and len(mask_rows) < args.mask_samples:
                batch_names = list(names) if isinstance(names, (list, tuple)) else [names] * images.shape[0]
                for sample_idx in range(images.shape[0]):
                    if len(mask_rows) >= args.mask_samples:
                        break
                    stem = "{:03d}_{}".format(len(mask_rows), _safe_stem(batch_names[sample_idx]))
                    image_cpu = images[sample_idx].detach().cpu()
                    pred_cpu = pred[sample_idx].detach().cpu()
                    label_cpu = labels[sample_idx].detach().cpu()
                    conf_cpu = conf[sample_idx].detach().cpu()
                    entropy_cpu = entropy[sample_idx].detach().cpu()
                    selected_cpu = selected[sample_idx].detach().cpu()

                    image_path = mask_dir / "{}_image.png".format(stem)
                    pred_path = mask_dir / "{}_pred.png".format(stem)
                    gt_path = mask_dir / "{}_gt.png".format(stem)
                    confidence_path = mask_dir / "{}_confidence.png".format(stem)
                    entropy_path = mask_dir / "{}_entropy.png".format(stem)
                    error_path = mask_dir / "{}_error.png".format(stem)
                    pl_selection_path = mask_dir / "{}_pl_selection.png".format(stem)
                    _image_to_pil(image_cpu, cfg).save(image_path)
                    _label_to_rgb(pred_cpu.numpy()).save(pred_path)
                    _label_to_rgb(label_cpu.numpy()).save(gt_path)

                    valid_sample = label_cpu != ignore
                    correct_sample = (pred_cpu == label_cpu) & valid_sample
                    selected_sample = selected_cpu & valid_sample
                    entropy_max = math.log(max(num_classes, 2))
                    _tensor_heatmap(conf_cpu, valid_sample, 0.0, 1.0).save(confidence_path)
                    _tensor_heatmap(entropy_cpu, valid_sample, 0.0, entropy_max).save(entropy_path)
                    _tensor_error_map(pred_cpu, label_cpu, valid_sample).save(error_path)
                    _pl_selection_map(selected_cpu, pred_cpu, label_cpu, valid_sample).save(pl_selection_path)

                    valid_pixels = max(float(valid_sample.sum().item()), 1.0)
                    selected_pixels_sample = float(selected_sample.sum().item())
                    sample_acc = float(((pred_cpu == label_cpu) & valid_sample).sum().item()) / max(
                        float(valid_sample.sum().item()),
                        1.0,
                    )
                    mask_rows.append(
                        {
                            "method": args.method,
                            "sample": stem,
                            "image": str(image_path),
                            "prediction": str(pred_path),
                            "ground_truth": str(gt_path),
                            "confidence_map": str(confidence_path),
                            "entropy_map": str(entropy_path),
                            "error_map": str(error_path),
                            "pl_selection_map": str(pl_selection_path),
                            "sample_miou": _sample_miou(pred_cpu, label_cpu, num_classes, ignore),
                            "sample_acc": sample_acc,
                            "mean_confidence": float(conf_cpu[valid_sample].mean().item())
                            if valid_sample.any()
                            else "",
                            "mean_entropy": float(entropy_cpu[valid_sample].mean().item())
                            if valid_sample.any()
                            else "",
                            "pl_coverage": selected_pixels_sample / valid_pixels,
                            "pl_precision": float((correct_sample & selected_sample).sum().item())
                            / max(selected_pixels_sample, 1.0),
                        }
                    )

    iou = intersection / (union + 1e-10)
    pl_iou = pl_intersection / (pl_union + 1e-10)
    groups = _class_groups(freq)
    ece = 0.0
    if ece_bins.sum() > 0:
        for idx in range(args.ece_bins):
            if ece_bins[idx] <= 0:
                continue
            avg_conf = ece_conf[idx] / ece_bins[idx]
            avg_acc = ece_acc[idx] / ece_bins[idx]
            ece += (ece_bins[idx] / ece_bins.sum()) * abs(avg_conf - avg_acc)

    class_rows = []
    for cls_id in range(num_classes):
        class_rows.append(
            {
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
                "pixels": int(freq[cls_id]),
                "iou": float(iou[cls_id]),
                "pl_iou_selected": float(pl_iou[cls_id]),
            }
        )

    payload = {
        "extra_analysis_status": "ok",
        "analysis_checkpoint": checkpoint,
        "analysis_batches": args.analysis_max_batches if args.analysis_max_batches > 0 else "all",
        "pl_threshold": args.pl_threshold,
        "pl_coverage": selected_pixels / max(total_valid, 1.0),
        "pl_precision": selected_correct / max(selected_pixels, 1.0),
        "pl_miou": _safe_mean(pl_iou),
        "ece": float(ece),
        "mean_entropy": entropy_sum / max(total_valid, 1.0),
        "confidence_gap": (conf_correct_sum / max(correct_count, 1.0))
        - (conf_wrong_sum / max(wrong_count, 1.0)),
        "bg_iou": float(iou[0]) if num_classes > 0 else math.nan,
        "foreground_miou": _safe_mean(iou[1:]),
        "rare_class_miou": _safe_mean(iou[groups["tail"]]) if groups["tail"] else math.nan,
        "head_miou": _safe_mean(iou[groups["head"]]) if groups["head"] else math.nan,
        "medium_miou": _safe_mean(iou[groups["medium"]]) if groups["medium"] else math.nan,
        "tail_miou": _safe_mean(iou[groups["tail"]]) if groups["tail"] else math.nan,
        "overall_miou": _safe_mean(iou),
        "tail_classes": ",".join(CLASS_NAMES.get(idx, str(idx)) for idx in groups["tail"]),
        "head_classes": ",".join(CLASS_NAMES.get(idx, str(idx)) for idx in groups["head"]),
        "class_metrics": class_rows,
        "mask_inference_dir": str(mask_dir) if save_masks else "",
        "mask_inference_samples": len(mask_rows),
    }
    if mask_rows:
        mask_csv = mask_dir / "mask_inference_samples.csv"
        with mask_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "method",
                    "sample",
                    "image",
                    "prediction",
                    "ground_truth",
                    "confidence_map",
                    "entropy_map",
                    "error_map",
                    "pl_selection_map",
                    "sample_miou",
                    "sample_acc",
                    "mean_confidence",
                    "mean_entropy",
                    "pl_coverage",
                    "pl_precision",
                ],
            )
            writer.writeheader()
            writer.writerows(mask_rows)
        payload["mask_inference_csv"] = str(mask_csv)

    rare_row = {name: payload.get(name, "") for name in RARE_CLASS_FIELDS}
    rare_row["method"] = args.method
    rare_row["method_label"] = METHOD_LABELS.get(args.method, args.method)
    _write_single_row_csv(out_dir / "rare_class_metrics.csv", rare_row, RARE_CLASS_FIELDS)
    append_csv_row(out_dir.parent / "rare_class_summary.csv", rare_row, RARE_CLASS_FIELDS)

    pseudo_row = {name: payload.get(name, "") for name in PSEUDO_QUALITY_FIELDS}
    pseudo_row["method"] = args.method
    pseudo_row["method_label"] = METHOD_LABELS.get(args.method, args.method)
    _write_single_row_csv(out_dir / "pseudo_label_quality.csv", pseudo_row, PSEUDO_QUALITY_FIELDS)
    append_csv_row(out_dir.parent / "pseudo_label_quality_summary.csv", pseudo_row, PSEUDO_QUALITY_FIELDS)

    write_json(out_dir / "extra_analysis.json", payload)
    del feature_extractor, classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


SUMMARY_FIELDS = [
    "method",
    "method_label",
    "train_script",
    "profile_batch_size",
    "paper_batch_size",
    "optimizer",
    "base_lr",
    "classifier_lr",
    "lr_method",
    "lr_power",
    "momentum",
    "weight_decay",
    "solver_stop_iter",
    "ema",
    "ema_update_period",
    "metapl_open_profile",
    "profile_snd_metapl",
    "method_hyperparams",
    "training_profile_status",
    "profile_batches",
    "synthetic_snd_meta_val",
    "train_elapsed_s_wall",
    "train_s_per_batch_wall",
    "train_s_per_iter_logged",
    "power_sampler",
    "train_energy_wh",
    "train_power_w_mean",
    "train_power_w_peak",
    "train_peak_cuda_allocated_mb_logged",
    "train_peak_gpu_memory_mb_sampled",
    "params_m",
    "trainable_params_m",
    "model_size_fp32_mb",
    "train_extra_networks",
    "train_forward_passes_est",
    "ops_status",
    "ops_tool",
    "ops_unit",
    "ops_g",
    "ops_m",
    "latency_ms_mean",
    "latency_ms_std",
    "fps",
    "peak_inference_memory_mb",
    "extra_module_at_test",
    "forward_passes_at_test",
    "validation_best_iter",
    "validation_best_miou",
    "extra_analysis_status",
    "pl_coverage",
    "pl_precision",
    "pl_miou",
    "ece",
    "mean_entropy",
    "confidence_gap",
    "bg_iou",
    "foreground_miou",
    "rare_class_miou",
    "head_miou",
    "medium_miou",
    "tail_miou",
    "overall_miou",
    "tail_classes",
    "analysis_checkpoint",
    "mask_inference_samples",
    "mask_inference_dir",
    "mask_inference_csv",
]

RARE_CLASS_FIELDS = [
    "method",
    "method_label",
    "bg_iou",
    "foreground_miou",
    "rare_class_miou",
    "head_miou",
    "medium_miou",
    "tail_miou",
    "overall_miou",
    "head_classes",
    "tail_classes",
    "analysis_checkpoint",
]

PSEUDO_QUALITY_FIELDS = [
    "method",
    "method_label",
    "pl_threshold",
    "pl_coverage",
    "pl_precision",
    "pl_miou",
    "ece",
    "mean_entropy",
    "confidence_gap",
    "analysis_checkpoint",
]

HYPERPARAMETER_FIELDS = [
    "method",
    "method_label",
    "method_family",
    "train_script",
    "config_file",
    "source_checkpoint",
    "source_train",
    "target_train",
    "test_set",
    "tt_set",
    "ii_set",
    "input_size",
    "profile_batches",
    "profile_batch_size",
    "profile_solver_max_iter",
    "profile_solver_stop_iter",
    "paper_preset",
    "paper_preset_source",
    "paper_batch_size",
    "optimizer",
    "base_lr",
    "classifier_lr",
    "classifier_lr_multiplier",
    "lr_method",
    "lr_power",
    "momentum",
    "weight_decay",
    "solver_max_iter",
    "solver_stop_iter",
    "checkpoint_period",
    "ema",
    "ema_update_period",
    "tcr_open",
    "tcr_update_frequency",
    "metapl_open",
    "metapl_open_profile",
    "profile_snd_metapl",
    "dtu_dynamic",
    "dtu_proxy_metric",
    "dtu_fix_iteration",
    "pl_threshold",
    "analysis_max_batches",
    "mask_samples",
    "train_extra_networks",
    "train_forward_passes_est",
    "method_hyperparams",
    "config_hyperparams",
    "script_hyperparams",
    "opts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile UIIS -> SUIM SFUDA efficiency.")
    parser.add_argument("--method", required=True, choices=sorted(BASELINE_METHODS | {"snd", "remask", "remask_dtst"}))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--config-file", default="configs/deeplabv2_r101_dtst_SUIM.yaml")
    parser.add_argument("--baseline-train-script", default="train_sfda_baselines.py")
    parser.add_argument("--snd-train-script", default="train_SND.py")
    parser.add_argument("--remask-train-script", default="train_ReMask_underwater.py")
    parser.add_argument("--source-checkpoint", default="source_model/src_uiis_to_suim_8c_b4_24k/model_iter018000.pth")
    parser.add_argument("--source-train", default="uiis_train")
    parser.add_argument("--target-train", default="suim_train")
    parser.add_argument("--test-set", default="suim_val")
    parser.add_argument("--tt-set", default="suim_TT_train")
    parser.add_argument("--ii-set", default="suim_II_train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--profile-batches", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--input-size", type=parse_size, default=parse_size("960x480"))
    parser.add_argument("--inference-batch-size", type=int, default=1)
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--latency-repeats", type=int, default=20)
    parser.add_argument("--measure-ops", action="store_true")
    parser.add_argument("--skip-training-profile", action="store_true")
    parser.add_argument("--paper-preset", action="store_true", default=True)
    parser.add_argument("--no-paper-preset", dest="paper_preset", action="store_false")
    parser.add_argument("--power-sample-interval", type=float, default=0.25)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--warm-artifact-dir", default="")
    parser.add_argument("--allow-candidate-build", action="store_true")
    parser.add_argument(
        "--profile-snd-metapl",
        action="store_true",
        help="Keep METAPL.OPEN=True when short-profiling SND so the meta-pseudo/bi-optimization path is measured.",
    )
    parser.add_argument("--extra-analysis", action="store_true")
    parser.add_argument("--strict-extra-analysis", action="store_true")
    parser.add_argument("--analysis-run-dir", default="")
    parser.add_argument("--analysis-checkpoint", default="")
    parser.add_argument("--analysis-max-batches", type=int, default=0)
    parser.add_argument("--analysis-batch-size", type=int, default=1)
    parser.add_argument("--pl-threshold", type=float, default=0.90)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--mask-samples", type=int, default=7)
    parser.add_argument("--no-mask-inference", action="store_true")
    add_underwater_label_args(parser)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    mkdir(str(out_dir))
    if args.profile_batches < 1:
        raise ValueError("--profile-batches must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    cudnn.benchmark = args.device == "cuda"

    started = time.time()
    result: Dict[str, Any] = {
        "method": args.method,
        "method_label": METHOD_LABELS.get(args.method, args.method),
        "output_dir": str(out_dir),
        "source_checkpoint": args.source_checkpoint,
        "config_file": args.config_file,
    }

    hyperparams = collect_hyperparameters(args, out_dir)
    result.update(hyperparams)
    write_json(out_dir / "hyperparameters.json", hyperparams)
    _write_single_row_csv(out_dir / "hyperparameters.csv", hyperparams, HYPERPARAMETER_FIELDS)
    hyperparameter_csv = out_dir.parent / "hyperparameter_summary.csv"
    append_csv_row(hyperparameter_csv, hyperparams, HYPERPARAMETER_FIELDS)

    training = run_training_profile(args, out_dir)
    result.update(training)

    static = profile_static_and_inference(args, out_dir)
    result.update(static)

    curve = validation_curve(args, out_dir)
    result.update(curve)

    extra = analyze_checkpoint(args, out_dir)
    for key, value in extra.items():
        if key != "class_metrics":
            result[key] = value

    result["total_profiler_elapsed_s"] = time.time() - started
    write_json(out_dir / "efficiency_profile.json", result)
    summary_csv = out_dir.parent / "efficiency_summary.csv"
    append_csv_row(summary_csv, result, SUMMARY_FIELDS)

    print(json.dumps({key: result.get(key, "") for key in SUMMARY_FIELDS}, indent=2, default=_as_jsonable))
    print("Wrote {}".format(out_dir / "efficiency_profile.json"))
    print("Wrote {}".format(out_dir / "hyperparameters.json"))
    print("Updated {}".format(summary_csv))
    print("Updated {}".format(hyperparameter_csv))


if __name__ == "__main__":
    main()
