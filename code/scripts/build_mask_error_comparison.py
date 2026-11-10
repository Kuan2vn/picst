#!/usr/bin/env python3
"""Build a result-consistent SND-vs-ReMask qualitative error panel.

The script only composes existing exported images and predictions. Sample IDs
are explicit so the selection rule is auditable and never depends on a newly
computed performance ranking.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SND_CSV = (
    "validate_results/efficiency_uiis_to_suim/"
    "remask_high_gap_26496_mask_export/snd_metapl/"
    "mask_inference/mask_inference_samples.csv"
)
DEFAULT_REMASK_CSV = (
    "validate_results/efficiency_uiis_to_suim/"
    "remask_high_gap_26496_mask_export/remask/"
    "mask_inference/mask_inference_samples.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose aligned SND/ReMask predictions and error maps."
    )
    parser.add_argument("--snd-csv", default=DEFAULT_SND_CSV)
    parser.add_argument("--remask-csv", default=DEFAULT_REMASK_CSV)
    parser.add_argument(
        "--samples",
        default="000_d_r_122,003_d_r_166,006_d_r_233",
        help="Comma-separated exact sample IDs. Defaults are fixed dataset-order IDs.",
    )
    parser.add_argument("--snd-label", default="SND")
    parser.add_argument("--remask-label", default="ReMask")
    parser.add_argument(
        "--output",
        default="validate_results/mask_analysis/snd_vs_remask_fixed_samples.png",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_rows(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["sample"]): row for row in rows}


def load_rgb(path_value: str, size=None, is_mask: bool = False) -> np.ndarray:
    path = resolve_path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        resampling = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
        image = image.resize(size, resampling)
    return np.asarray(image)


def error_mask(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    error = np.any(prediction != ground_truth, axis=2)
    ignored = np.all(ground_truth == np.asarray((128, 128, 128), dtype=np.uint8), axis=2)
    output = error.astype(np.float32)
    output[ignored] = 0.5
    return output


def error_change(
    snd_error: np.ndarray,
    remask_error: np.ndarray,
) -> np.ndarray:
    """RGB: green=corrected, red=new error, white=both wrong, black=both correct."""
    result = np.zeros((*snd_error.shape, 3), dtype=np.uint8)
    snd_wrong = snd_error > 0.75
    remask_wrong = remask_error > 0.75
    ignored = (snd_error == 0.5) | (remask_error == 0.5)
    result[snd_wrong & remask_wrong] = (230, 230, 230)
    result[snd_wrong & ~remask_wrong] = (35, 180, 80)
    result[~snd_wrong & remask_wrong] = (220, 55, 55)
    result[ignored] = (128, 128, 128)
    return result


def validate_samples(
    samples: Iterable[str],
    snd_rows: Dict[str, Dict[str, str]],
    remask_rows: Dict[str, Dict[str, str]],
) -> List[str]:
    selected = []
    for sample in samples:
        sample = sample.strip()
        if not sample:
            continue
        if sample not in snd_rows or sample not in remask_rows:
            raise KeyError("Sample '{}' is not present in both CSV files.".format(sample))
        selected.append(sample)
    if not selected:
        raise ValueError("No samples selected.")
    return selected


def main() -> None:
    args = parse_args()
    snd_csv = resolve_path(args.snd_csv)
    remask_csv = resolve_path(args.remask_csv)
    snd_rows = read_rows(snd_csv)
    remask_rows = read_rows(remask_csv)
    samples = validate_samples(args.samples.split(","), snd_rows, remask_rows)

    fig, axes = plt.subplots(len(samples), 7, figsize=(15.8, 2.55 * len(samples)))
    if len(samples) == 1:
        axes = axes[np.newaxis, :]
    column_titles = (
        "Image",
        "Ground Truth",
        args.snd_label,
        args.remask_label,
        "SND error",
        "ReMask error",
        "Error change",
    )

    for row_index, sample in enumerate(samples):
        snd_row = snd_rows[sample]
        remask_row = remask_rows[sample]
        image_pil = Image.open(resolve_path(remask_row["image"])).convert("RGB")
        size = image_pil.size
        image = np.asarray(image_pil)
        ground_truth = load_rgb(remask_row["ground_truth"], size=size, is_mask=True)
        snd_pred = load_rgb(snd_row["prediction"], size=size, is_mask=True)
        remask_pred = load_rgb(remask_row["prediction"], size=size, is_mask=True)
        snd_error = error_mask(snd_pred, ground_truth)
        remask_error = error_mask(remask_pred, ground_truth)
        change = error_change(snd_error, remask_error)

        panels = (
            (image, None),
            (ground_truth, None),
            (snd_pred, None),
            (remask_pred, None),
            (snd_error, "gray"),
            (remask_error, "gray"),
            (change, None),
        )
        for column_index, (values, cmap) in enumerate(panels):
            ax = axes[row_index, column_index]
            if cmap is None:
                ax.imshow(values, interpolation="nearest")
            else:
                ax.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
            if row_index == 0:
                ax.set_title(column_titles[column_index], fontsize=11, fontweight="semibold")
            if column_index == 0:
                ax.set_ylabel(sample, fontsize=9, rotation=90, labelpad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    legend = (
        Patch(facecolor=(35 / 255, 180 / 255, 80 / 255), label="Corrected by ReMask"),
        Patch(facecolor=(220 / 255, 55 / 255, 55 / 255), label="New ReMask error"),
        Patch(facecolor=(230 / 255, 230 / 255, 230 / 255), edgecolor="0.5", label="Both wrong"),
        Patch(facecolor="black", label="Both correct"),
    )
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        fontsize=9,
        frameon=False,
    )
    fig.subplots_adjust(left=0.02, right=0.995, top=0.94, bottom=0.09, wspace=0.025, hspace=0.06)

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(str(output.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote {}".format(output.resolve()))
    print("Wrote {}".format(output.with_suffix(".pdf").resolve()))


if __name__ == "__main__":
    sys.exit(main())
