#!/usr/bin/env python3
"""Select high-gap ReMask qualitative samples from mask_inference outputs.

The script is intentionally post-hoc: it does not run model inference. It ranks
the images already present in mask_inference_samples.csv by

    ReMask sample mIoU - best non-ReMask sample mIoU.

If the input folders contain only seven mask samples, the ranking is over those
seven examples. If HPC profiling is rerun with a large MASK_SAMPLES value, the
same script will select the top-gap examples from that larger candidate pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFont


METHOD_LABELS = {
    "tt_sfuda": "TT-SFUDA",
    "cal_sfda": "Cal-SFDA",
    "sfmim": "SFMIM",
    "crossmatch": "CrossMatch",
    "uotsl": "UOTSL",
    "iapc": "IAPC",
    "snd": "SND",
    "snd_core": "SND",
    "snd_metapl": "SND",
    "remask": "ReMask",
    "remask_dtst": "ReMask",
}

DEFAULT_BASELINES = [
    "tt_sfuda",
    "cal_sfda",
    "sfmim",
    "crossmatch",
    "uotsl",
    "iapc",
    "snd_metapl",
]

DEFAULT_PANEL_METHODS = ["cal_sfda", "crossmatch", "iapc", "snd_metapl", "remask"]


def as_float(value: object, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(path_text: str, repo_root: Path, csv_dir: Path) -> Path:
    path = Path(path_text)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([repo_root / path, csv_dir / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path


def normalize_key(sample: str, image_path: str = "") -> str:
    """Match samples even if different runs use different numeric prefixes."""
    base = sample.strip()
    if not base and image_path:
        base = Path(image_path).name
        base = re.sub(r"_(image|pred|gt)\.[^.]+$", "", base)
    return re.sub(r"^\d+_", "", base)


def method_dir(profile_root: Path, method: str) -> Optional[Path]:
    direct = profile_root / method
    if (direct / "mask_inference" / "mask_inference_samples.csv").is_file():
        return direct
    aliases = {
        "snd": ["snd_metapl", "snd_core"],
        "remask": ["remask_dtst"],
    }
    for alias in aliases.get(method, []):
        candidate = profile_root / alias
        if (candidate / "mask_inference" / "mask_inference_samples.csv").is_file():
            return candidate
    return None


def load_method_rows(profile_dir: Path, method_id: str, repo_root: Path) -> Dict[str, Dict[str, object]]:
    csv_path = profile_dir / "mask_inference" / "mask_inference_samples.csv"
    rows: Dict[str, Dict[str, object]] = {}
    for row in read_csv(csv_path):
        key = normalize_key(row.get("sample", ""), row.get("image", ""))
        if not key:
            continue
        csv_dir = csv_path.parent
        enriched: Dict[str, object] = dict(row)
        enriched["method_id"] = method_id
        enriched["method_label"] = METHOD_LABELS.get(method_id, method_id)
        enriched["sample_key"] = key
        enriched["sample_miou_float"] = as_float(row.get("sample_miou"))
        for field in (
            "image",
            "prediction",
            "ground_truth",
            "confidence_map",
            "entropy_map",
            "error_map",
            "pl_selection_map",
        ):
            if row.get(field):
                enriched[field + "_path"] = resolve_path(row[field], repo_root, csv_dir)
        rows[key] = enriched
    return rows


def discover_rows(
    profile_root: Path,
    remask_profile_dir: Path,
    baselines: Sequence[str],
    repo_root: Path,
) -> Tuple[Dict[str, Dict[str, Dict[str, object]]], List[str]]:
    by_method: Dict[str, Dict[str, Dict[str, object]]] = {}
    missing: List[str] = []
    seen_csvs = set()
    for method in baselines:
        profile_dir = method_dir(profile_root, method)
        if profile_dir is None:
            missing.append(method)
            continue
        csv_path = (profile_dir / "mask_inference" / "mask_inference_samples.csv").resolve()
        if csv_path in seen_csvs:
            continue
        seen_csvs.add(csv_path)
        by_method[method] = load_method_rows(profile_dir, method, repo_root)

    if not (remask_profile_dir / "mask_inference" / "mask_inference_samples.csv").is_file():
        raise FileNotFoundError(
            "Missing ReMask mask_inference_samples.csv under {}".format(remask_profile_dir)
        )
    by_method["remask"] = load_method_rows(remask_profile_dir, "remask", repo_root)
    return by_method, missing


def rank_samples(
    by_method: Dict[str, Dict[str, Dict[str, object]]],
    baselines: Sequence[str],
    min_gap: float,
    require_positive: bool,
) -> List[Dict[str, object]]:
    remask_rows = by_method.get("remask", {})
    ranking: List[Dict[str, object]] = []
    for key, remask_row in remask_rows.items():
        remask_miou = as_float(remask_row.get("sample_miou_float"))
        if math.isnan(remask_miou):
            continue
        baseline_scores: List[Tuple[str, float]] = []
        for method in baselines:
            row = by_method.get(method, {}).get(key)
            if not row:
                continue
            score = as_float(row.get("sample_miou_float"))
            if not math.isnan(score):
                baseline_scores.append((method, score))
        if not baseline_scores:
            continue
        best_method, best_score = max(baseline_scores, key=lambda item: item[1])
        mean_baseline = sum(score for _, score in baseline_scores) / len(baseline_scores)
        gap = remask_miou - best_score
        if require_positive and gap <= 0:
            continue
        if gap < min_gap:
            continue
        ranking.append(
            {
                "sample_key": key,
                "sample": remask_row.get("sample", key),
                "remask_miou": remask_miou,
                "best_baseline": best_method,
                "best_baseline_label": METHOD_LABELS.get(best_method, best_method),
                "best_baseline_miou": best_score,
                "gap_vs_best": gap,
                "mean_baseline_miou": mean_baseline,
                "gap_vs_mean": remask_miou - mean_baseline,
                "available_baselines": " ".join(method for method, _ in baseline_scores),
                "num_baselines": len(baseline_scores),
            }
        )
    ranking.sort(key=lambda row: (as_float(row["gap_vs_best"]), as_float(row["remask_miou"])), reverse=True)
    return ranking


def load_image(path: Optional[Path], fallback_size: Tuple[int, int] = (960, 480)) -> Image.Image:
    if path and path.is_file():
        return Image.open(path).convert("RGB")
    return Image.new("RGB", fallback_size, (245, 245, 245))


def fit_canvas(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    img = image.copy()
    img.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_header(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(248, 248, 248), outline=(220, 220, 220))
    wrapped = "\n".join(textwrap.wrap(text, width=max(10, (x1 - x0) // 12))[:2])
    draw.multiline_text((x0 + 8, y0 + 6), wrapped, fill=(20, 20, 20), font=font(16), spacing=2)


def get_path(row: Dict[str, object], field: str) -> Optional[Path]:
    value = row.get(field + "_path")
    return value if isinstance(value, Path) else None


def error_map(pred: Image.Image, gt: Image.Image) -> Image.Image:
    pred = pred.convert("RGB")
    gt = gt.convert("RGB")
    if pred.size != gt.size:
        gt = gt.resize(pred.size, Image.Resampling.NEAREST)
    diff = ImageChops.difference(pred, gt).convert("L")
    mask = diff.point(lambda px: 255 if px > 0 else 0)
    correct = Image.new("RGB", pred.size, (210, 235, 210))
    wrong = Image.new("RGB", pred.size, (225, 70, 70))
    correct.paste(wrong, mask=mask)
    return correct


def text_panel(size: Tuple[int, int], lines: Sequence[str], title_size: int = 22, body_size: int = 18) -> Image.Image:
    panel = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(panel)
    y = 18
    for idx, line in enumerate(lines):
        if not line:
            y += 12
            continue
        wrapped = textwrap.wrap(line, width=max(18, size[0] // 11)) or [""]
        for part in wrapped:
            draw.text(
                (18, y),
                part,
                fill=(20, 20, 20),
                font=font(title_size if idx == 0 else body_size),
            )
            y += title_size + 8 if idx == 0 else body_size + 8
    return panel


def optional_map_panel(
    row: Dict[str, object],
    field: str,
    title: str,
    image_size: Tuple[int, int],
    thumb_size: Tuple[int, int],
) -> Image.Image:
    path = get_path(row, field)
    if path and path.is_file():
        return load_image(path, image_size)
    return text_panel(
        thumb_size,
        [
            title,
            "Not exported in this profile.",
            "Rerun this Slurm with RERUN_MASK_EXPORT=1 to generate it.",
        ],
        title_size=18,
        body_size=14,
    )


def make_method_panel(
    sample_key: str,
    rows_for_methods: Sequence[Tuple[str, Dict[str, object]]],
    out_path: Path,
    ranking_row: Dict[str, object],
    thumb_size: Tuple[int, int],
) -> None:
    remask_row = dict(rows_for_methods[-1][1])
    image = load_image(get_path(remask_row, "image"))
    gt = load_image(get_path(remask_row, "ground_truth"), image.size)
    columns: List[Tuple[str, Image.Image]] = [("Image", image), ("GT", gt)]
    for method, row in rows_for_methods:
        pred = load_image(get_path(row, "prediction"), image.size)
        miou = as_float(row.get("sample_miou_float"))
        label = METHOD_LABELS.get(method, method)
        if not math.isnan(miou):
            label = "{}\nmIoU {:.3f}".format(label, miou)
        columns.append((label, pred))

    header_h = 52
    w = thumb_size[0] * len(columns)
    h = header_h + thumb_size[1] + 72
    panel = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(panel)
    for idx, (label, img) in enumerate(columns):
        x0 = idx * thumb_size[0]
        draw_header(draw, (x0, 0, x0 + thumb_size[0], header_h), label)
        panel.paste(fit_canvas(img, thumb_size), (x0, header_h))

    caption = (
        "sample={} | ReMask-best gap={:.4f} | best baseline={} ({:.4f})".format(
            sample_key,
            as_float(ranking_row.get("gap_vs_best")),
            ranking_row.get("best_baseline_label", ""),
            as_float(ranking_row.get("best_baseline_miou")),
        )
    )
    draw.rectangle((0, header_h + thumb_size[1], w, h), fill=(255, 255, 255))
    draw.text((10, header_h + thumb_size[1] + 12), caption, fill=(20, 20, 20), font=font(18))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def make_remask_diagnostic(
    sample_key: str,
    remask_row: Dict[str, object],
    ranking_row: Dict[str, object],
    out_path: Path,
    thumb_size: Tuple[int, int],
) -> None:
    image = load_image(get_path(remask_row, "image"))
    gt = load_image(get_path(remask_row, "ground_truth"), image.size)
    pred = load_image(get_path(remask_row, "prediction"), image.size)
    err_path = get_path(remask_row, "error_map")
    if err_path and err_path.is_file():
        err = load_image(err_path, image.size)
    else:
        err = error_map(pred, gt)
    confidence = optional_map_panel(remask_row, "confidence_map", "Confidence", image.size, thumb_size)
    entropy = optional_map_panel(remask_row, "entropy_map", "Entropy", image.size, thumb_size)
    pl_selection = optional_map_panel(remask_row, "pl_selection_map", "PL selection", image.size, thumb_size)

    lines = [
        "Sample: {}".format(sample_key),
        "ReMask mIoU: {:.4f}".format(as_float(ranking_row.get("remask_miou"))),
        "Best baseline: {} ({:.4f})".format(
            ranking_row.get("best_baseline_label", ""),
            as_float(ranking_row.get("best_baseline_miou")),
        ),
        "Gap vs best: +{:.4f}".format(as_float(ranking_row.get("gap_vs_best"))),
        "Gap vs mean baseline: +{:.4f}".format(as_float(ranking_row.get("gap_vs_mean"))),
        "Mean conf: {:.4f}".format(as_float(remask_row.get("mean_confidence"))),
        "Mean entropy: {:.4f}".format(as_float(remask_row.get("mean_entropy"))),
        "PL cov/prec: {:.4f}/{:.4f}".format(
            as_float(remask_row.get("pl_coverage")),
            as_float(remask_row.get("pl_precision")),
        ),
        "",
        "Error: green correct, red mismatch.",
        "PL map: green selected-correct, red selected-wrong, blue rejected.",
    ]
    metrics = text_panel(thumb_size, lines, title_size=18, body_size=13)

    columns = [
        ("Image", image),
        ("GT", gt),
        ("ReMask pred", pred),
        ("ReMask error", err),
        ("Confidence", confidence),
        ("Entropy", entropy),
        ("PL selection", pl_selection),
        ("Metrics", metrics),
    ]
    header_h = 52
    grid_cols = 4
    grid_rows = int(math.ceil(len(columns) / float(grid_cols)))
    w = thumb_size[0] * grid_cols
    h = (header_h + thumb_size[1]) * grid_rows
    panel = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(panel)
    for idx, (label, img) in enumerate(columns):
        row = idx // grid_cols
        col = idx % grid_cols
        x0 = col * thumb_size[0]
        y0 = row * (header_h + thumb_size[1])
        draw_header(draw, (x0, y0, x0 + thumb_size[0], y0 + header_h), label)
        panel.paste(fit_canvas(img, thumb_size), (x0, y0 + header_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def copy_raw_sample(sample_key: str, rows: Sequence[Tuple[str, Dict[str, object]]], out_dir: Path) -> None:
    sample_dir = out_dir / sample_key
    sample_dir.mkdir(parents=True, exist_ok=True)
    for method, row in rows:
        for field, suffix in (
            ("prediction", "pred"),
            ("image", "image"),
            ("ground_truth", "gt"),
            ("confidence_map", "confidence"),
            ("entropy_map", "entropy"),
            ("error_map", "error"),
            ("pl_selection_map", "pl_selection"),
        ):
            src = get_path(row, field)
            if src and src.is_file():
                shutil.copy2(src, sample_dir / "{}_{}.png".format(method, suffix))


def panel_methods_for_row(panel_methods: Sequence[str], best_baseline: object) -> List[str]:
    methods: List[str] = []
    for method in panel_methods:
        if method not in methods:
            methods.append(method)
    best = str(best_baseline or "")
    if best and best not in methods:
        if "remask" in methods:
            methods.insert(methods.index("remask"), best)
        else:
            methods.append(best)
    if "remask" in methods:
        methods = [method for method in methods if method != "remask"] + ["remask"]
    else:
        methods.append("remask")
    return methods


def write_report(path: Path, args: argparse.Namespace, ranking: Sequence[Dict[str, object]], missing: Sequence[str]) -> None:
    lines = [
        "# ReMask high-gap qualitative selection",
        "",
        "- Baseline profile root: `{}`".format(args.profile_root),
        "- ReMask profile dir: `{}`".format(args.remask_profile_dir),
        "- Candidate samples with ReMask and at least one baseline: `{}`".format(len(ranking)),
        "- Selected top-k: `{}`".format(min(args.top_k, len(ranking))),
        "- Missing baseline folders: `{}`".format(" ".join(missing) if missing else "none"),
        "",
        "This is a post-hoc selector over existing `mask_inference_samples.csv` files. "
        "To select from the full validation set, first generate mask samples for all target images.",
        "When the profiler is rerun with the updated exporter, ReMask diagnostics include confidence, "
        "entropy, prediction-error, and pseudo-label selection maps.",
        "",
        "## Top samples",
        "",
        "| Rank | Sample | ReMask | Best baseline | Gap | Mean baseline | Gap mean |",
        "|---:|---|---:|---|---:|---:|---:|",
    ]
    for idx, row in enumerate(ranking[: args.top_k], 1):
        lines.append(
            "| {} | {} | {:.4f} | {} {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                idx,
                row.get("sample_key", ""),
                as_float(row.get("remask_miou")),
                row.get("best_baseline_label", ""),
                as_float(row.get("best_baseline_miou")),
                as_float(row.get("gap_vs_best")),
                as_float(row.get("mean_baseline_miou")),
                as_float(row.get("gap_vs_mean")),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True, help="Root containing baseline method profile folders.")
    parser.add_argument("--remask-profile-dir", type=Path, default=None, help="ReMask profile folder. Defaults to profile-root/remask.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-gap", type=float, default=-1.0)
    parser.add_argument("--allow-negative-gap", action="store_true")
    parser.add_argument("--baseline-methods", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--panel-methods", nargs="+", default=DEFAULT_PANEL_METHODS)
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--thumb-height", type=int, default=180)
    parser.add_argument("--no-copy-raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    profile_root = args.profile_root.resolve()
    remask_profile_dir = (args.remask_profile_dir or (profile_root / "remask")).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    by_method, missing = discover_rows(profile_root, remask_profile_dir, args.baseline_methods, repo_root)
    ranking = rank_samples(
        by_method,
        args.baseline_methods,
        min_gap=args.min_gap,
        require_positive=not args.allow_negative_gap,
    )

    fields = [
        "sample_key",
        "sample",
        "remask_miou",
        "best_baseline",
        "best_baseline_label",
        "best_baseline_miou",
        "gap_vs_best",
        "mean_baseline_miou",
        "gap_vs_mean",
        "available_baselines",
        "num_baselines",
    ]
    write_csv(output_dir / "high_gap_ranking.csv", ranking, fields)

    selected = ranking[: args.top_k]
    thumb_size = (args.thumb_width, args.thumb_height)
    comparison_dir = output_dir / "method_comparison_panels"
    diagnostic_dir = output_dir / "remask_diagnostics"
    raw_dir = output_dir / "selected_raw"

    for idx, row in enumerate(selected, 1):
        key = str(row["sample_key"])
        rows_for_panel: List[Tuple[str, Dict[str, object]]] = []
        panel_methods = panel_methods_for_row(args.panel_methods, row.get("best_baseline"))
        for method in panel_methods:
            method_key = "remask" if method in {"remask", "remask_dtst"} else method
            method_rows = by_method.get(method_key, {})
            if key in method_rows:
                rows_for_panel.append((method_key, method_rows[key]))
        if "remask" not in [method for method, _ in rows_for_panel] and key in by_method["remask"]:
            rows_for_panel.append(("remask", by_method["remask"][key]))
        if not rows_for_panel:
            continue

        panel_path = comparison_dir / "{:02d}_{}_comparison.png".format(idx, key)
        make_method_panel(key, rows_for_panel, panel_path, row, thumb_size)

        remask_row = by_method["remask"][key]
        diag_path = diagnostic_dir / "{:02d}_{}_remask_diagnostic.png".format(idx, key)
        make_remask_diagnostic(key, remask_row, row, diag_path, thumb_size)

        if not args.no_copy_raw:
            raw_rows = []
            for method in set(args.panel_methods + [str(row.get("best_baseline", "")), "remask"]):
                method_key = "remask" if method in {"remask", "remask_dtst"} else method
                method_row = by_method.get(method_key, {}).get(key)
                if method_row:
                    raw_rows.append((method_key, method_row))
            copy_raw_sample(key, raw_rows, raw_dir)

    write_report(output_dir / "README.md", args, ranking, missing)
    metadata = {
        "profile_root": str(profile_root),
        "remask_profile_dir": str(remask_profile_dir),
        "output_dir": str(output_dir),
        "candidate_count": len(ranking),
        "selected_count": len(selected),
        "missing_baselines": list(missing),
        "baseline_methods": list(args.baseline_methods),
        "panel_methods": list(args.panel_methods),
    }
    (output_dir / "selection_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Selected {} high-gap ReMask samples from {} candidates.".format(len(selected), len(ranking)))
    print("Output:", output_dir)


if __name__ == "__main__":
    main()
