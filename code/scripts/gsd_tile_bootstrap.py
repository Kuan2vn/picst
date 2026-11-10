"""Paired tile bootstrap of fixed class-coverage selectors using a saved score cache."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def selector_counts(score, labels, error, tile_ids, coverage, tiles):
    selected = np.zeros((tiles, 6), dtype=np.int64)
    wrong = np.zeros_like(selected)
    for label in range(6):
        indices = np.flatnonzero(labels == label)
        retained = math.ceil(coverage * len(indices))
        chosen = indices[np.argsort(-score[indices], kind="stable")[:retained]]
        selected[:, label] = np.bincount(tile_ids[chosen], minlength=tiles)
        wrong[:, label] = np.bincount(tile_ids[chosen], weights=error[chosen], minlength=tiles).astype(np.int64)
    return selected, wrong


def risks(selected, wrong):
    micro = wrong.sum(-1) / selected.sum(-1)
    by_class = np.divide(wrong, selected, out=np.full(wrong.shape, np.nan), where=selected > 0)
    macro = np.nansum(by_class, axis=-1) / np.isfinite(by_class).sum(-1)
    return micro, macro


def analyze(cache_path, output, coverage=.5, penalty=1., repeats=2000, seed=42):
    if not 0 < coverage <= 1 or penalty < 0 or not math.isfinite(penalty) or repeats < 100:
        raise ValueError("Invalid coverage, penalty or bootstrap count")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(cache_path, allow_pickle=False) as cache:
        labels, truth = cache["native_label"], cache["ground_truth"]
        eligible = cache["eligible"].astype(bool) & (truth != 255)
        names = cache["names"].astype(str)
        if labels.shape != truth.shape or eligible.shape != labels.shape or len(names) != len(labels):
            raise ValueError("Cache names, labels, truth and support have different grids")
        parent_names = ["_".join(Path(name).stem.split("_")[:2]) for name in names]
        tiles = sorted(set(parent_names))
        if len(tiles) < 2 or not eligible.any():
            raise ValueError("Bootstrap requires at least two parent tiles and labeled support")
        tile_map = {name: index for index, name in enumerate(tiles)}
        ids = np.asarray([tile_map[name] for name in parent_names], dtype=np.int32)
        tile_ids = np.broadcast_to(ids[:, None, None], labels.shape)[eligible]
        candidate = labels[eligible]
        if not np.all((candidate >= 0) & (candidate < 6)):
            raise ValueError("Candidate labels must be in classes 0..5")
        error = candidate != truth[eligible]
        confidence = cache["confidence"][eligible]
        weights = np.random.default_rng(seed).multinomial(len(tiles), np.ones(len(tiles))/len(tiles), repeats)
        point, samples, rows = {}, {}, []
        views = sorted(key[3:] for key in cache.files if key.startswith("js_"))
        for view in ["confidence", *views]:
            score = confidence if view == "confidence" else confidence * np.exp(-penalty * cache["js_" + view][eligible])
            selected, wrong = selector_counts(score, candidate, error, tile_ids, coverage, len(tiles))
            micro, macro = risks(selected.sum(0), wrong.sum(0))
            sampled_micro, sampled_macro = risks(weights @ selected, weights @ wrong)
            point[view] = {"micro_risk": float(micro), "macro_predicted_class_risk": float(macro),
                           "micro_ci95": np.quantile(sampled_micro, [.025, .975]).tolist(),
                           "macro_ci95": np.quantile(sampled_macro, [.025, .975]).tolist(),
                           "selected": int(selected.sum()), "wrong": int(wrong.sum())}
            samples[view] = sampled_micro
            for index, tile in enumerate(tiles):
                rows.append(f"{view},{tile},{int(selected[index].sum())},{int(wrong[index].sum())}")
        differences = {}
        for view in views:
            for control in ["confidence", *[v for v in views if v != view]]:
                delta = samples[view] - samples[control]
                differences[f"{view}_minus_{control}"] = {
                    "micro_risk_difference": point[view]["micro_risk"] - point[control]["micro_risk"],
                    "ci95": np.quantile(delta, [.025, .975]).tolist()}
    metadata_path = cache_path.parent / "metadata.json"
    result = {"definition": "Paired parent-tile bootstrap, conditional on original score cutoffs; no ranking refit",
              "scope": "Development pseudo-label risk, not adaptation performance or final benchmark confidence intervals",
              "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
              "cache_metadata": json.loads(metadata_path.read_text()) if metadata_path.is_file() else None,
              "parent_tiles": tiles, "patches": len(names), "coverage": coverage, "lambda": penalty,
              "bootstrap_repeats": repeats, "seed": seed, "selectors": point, "paired_differences": differences}
    (output / "bootstrap_report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (output / "per_tile_selected_risk.csv").write_text("selector,parent_tile,selected,wrong\n" + "\n".join(rows) + "\n")
    print(json.dumps({"output": str(output), "tiles": len(tiles), "coverage": coverage, "lambda": penalty}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=float, default=.5)
    parser.add_argument("--penalty", type=float, default=1.)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(args.cache, args.output, args.coverage, args.penalty, args.repeats, args.seed)
