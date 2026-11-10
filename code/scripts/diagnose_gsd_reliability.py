"""Frozen-teacher inference, matched class coverage, and reusable CPU score sweeps."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from scripts.gsd_reliability import normalized_js, warp_to_native
from core.models.segformer import build_segformer_b2


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def tensor(path):
    with Image.open(path) as image:
        values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    values = values.transpose(2, 0, 1)
    values = (values - np.array([.485, .456, .406])[:, None, None]) / np.array([.229, .224, .225])[:, None, None]
    return torch.from_numpy(values.astype(np.float32))[None]


def coverage_risks(score, labels, error, coverages):
    """One stable ordering per predicted class, with fixed native candidate labels."""
    cumulative, counts = {}, {}
    for label in range(6):
        indices = np.flatnonzero(labels == label)
        order = indices[np.argsort(-score[indices], kind="stable")]
        cumulative[label] = np.cumsum(error[order], dtype=np.int64)
        counts[label] = len(order)
    results = []
    for coverage in coverages:
        selected, wrong, class_risk, selected_by_class = 0, 0, {}, {}
        for label, count in counts.items():
            retained = math.ceil(coverage * count)
            mistakes = int(cumulative[label][retained - 1]) if retained else 0
            selected += retained
            wrong += mistakes
            selected_by_class[str(label)] = retained
            class_risk[str(label)] = mistakes / retained if retained else None
        present = [x for x in class_risk.values() if x is not None]
        results.append({"coverage": coverage, "selected": selected, "wrong": wrong,
            "micro_risk": wrong / selected if selected else None,
            "macro_predicted_class_risk": float(np.mean(present)) if present else None,
            "risk_by_predicted_class": class_risk, "selected_by_class": selected_by_class})
    return results


def per_image_risk(score, labels, truth, eligible, coverage=.5):
    # Rank without looking at GT; GT ignore is applied only to error reporting.
    total = np.zeros(6, dtype=np.int64)
    wrong = np.zeros(6, dtype=np.int64)
    for image in range(len(labels)):
        native, values = labels[image].flatten(), score[image].flatten()
        support, gt = eligible[image].flatten(), truth[image].flatten()
        for label in range(6):
            indices = np.flatnonzero(support & (native == label))
            retained = math.ceil(coverage*len(indices))
            selected = indices[np.argsort(-values[indices], kind='stable')[:retained]]
            selected = selected[gt[selected] != 255]
            total[label] += len(selected)
            wrong[label] += int((native[selected] != gt[selected]).sum())
    return {'coverage': coverage, 'selected_gt_valid': total.tolist(),
            'wrong': wrong.tolist(), 'micro_risk': float(wrong.sum()/max(1, total.sum())),
            'precision_by_predicted_class': [float(1-w/n) if n else None for w, n in zip(wrong, total)],
            'ranking': 'per image and predicted class; GT never enters the ordering/selection'}


def analyze(cache_path, output, lambdas=(0., .5, 1., 2.), previous_cache=None):
    output.mkdir(parents=True, exist_ok=True)
    cache = np.load(cache_path)
    labels, truth = cache["native_label"], cache["ground_truth"]
    eligible = cache["eligible"].astype(bool) & (truth != 255)
    if not eligible.any():
        raise ValueError("Diagnostic has no eligible labeled pixels")
    conf = cache["confidence"][eligible]
    native = labels[eligible]
    error = (native != truth[eligible])
    coverages = (.1, .25, .5, .75, .9)
    result = {"candidate_labels": "native, identical for every selector",
              "evaluation_support": "common geometric support intersected with GT != ignore255; evaluation only",
              "valid_labeled_pixels": int(eligible.sum()), "native_error": float(error.mean()),
              "confidence": coverage_risks(conf, native, error, coverages), "views": {}}
    result['per_image_q0p5'] = {'confidence': per_image_risk(
        cache['confidence'], labels, truth, cache['eligible'].astype(bool))}
    def confusion(prediction):
        matrix = np.bincount((truth[eligible].astype(np.int64)*6+prediction[eligible]).flatten(), minlength=36).reshape(6, 6)
        tp = np.diag(matrix)
        union = matrix.sum(0)+matrix.sum(1)-tp
        return {'confusion_matrix': matrix.tolist(),
                'iou': [float(a/b) if b else None for a, b in zip(tp, union)]}
    result['native_metrics_on_common_support'] = confusion(labels)
    for key in sorted(k for k in cache.files if k.startswith("js_")):
        view = key[3:]
        divergence = cache[key][eligible]
        aligned = cache["aligned_" + view][eligible]
        aligned_correct = aligned == truth[eligible]
        native_correct = ~error
        view_result = {"js_mean": float(divergence.mean()), "lambda_sweep": {},
            "native_correct_aligned_correct": int((native_correct & aligned_correct).sum()),
            "native_correct_aligned_wrong": int((native_correct & ~aligned_correct).sum()),
            "native_wrong_aligned_correct": int((~native_correct & aligned_correct).sum()),
            "both_wrong_same_label": int((~native_correct & ~aligned_correct & (native == aligned)).sum()),
            "both_wrong_different_label": int((~native_correct & ~aligned_correct & (native != aligned)).sum())}
        for penalty in lambdas:
            score = conf * np.exp(-penalty * divergence)
            view_result["lambda_sweep"][str(penalty)] = coverage_risks(score, native, error, coverages)
        result['per_image_q0p5'][view] = per_image_risk(
            cache['confidence']*np.exp(-cache[key]), labels, truth, cache['eligible'].astype(bool))
        bins = []
        for label in range(6):
            for lower, upper in ((0., .5), (.5, .8), (.8, .95), (.95, 1.00001)):
                indices = (native == label) & (conf >= lower) & (conf < upper)
                if indices.sum() < 20:
                    continue
                values, mistakes = divergence[indices], error[indices]
                order = np.argsort(values, kind="stable")
                half = len(order) // 2
                bins.append({"class": label, "confidence_bin": [lower, min(upper, 1.)],
                    "count": len(order), "low_js_risk": float(mistakes[order[:half]].mean()),
                    "high_js_risk": float(mistakes[order[half:]].mean())})
        view_result["conditional_risk"] = bins
        view_result['aligned_metrics_on_common_support'] = confusion(cache['aligned_'+view])
        same_wrong = ~native_correct & ~aligned_correct & (native == aligned)
        view_result['same_wrong_by_native_class'] = np.bincount(native[same_wrong], minlength=6).tolist()
        result["views"][view] = view_result
    if previous_cache:
        current_meta = json.loads((Path(cache_path).parent / "metadata.json").read_text())
        previous_meta = json.loads((Path(previous_cache).parent / "metadata.json").read_text())
        source_hashes = {previous_meta.get("checkpoint_sha256"), previous_meta.get("original_checkpoint_sha256")}
        source_hashes.discard(None)
        if (previous_meta.get("stage") != "source" or
                current_meta.get("source_checkpoint_sha256") not in source_hashes or
                current_meta.get("view_manifest_sha256") != previous_meta.get("view_manifest_sha256")):
            raise ValueError("Temporal diagnostic requires a source cache and a descendant checkpoint on identical data")
        previous = np.load(previous_cache)
        if not np.array_equal(previous["names"], cache["names"]) or previous["native_label"].shape != labels.shape:
            raise ValueError("Temporal caches must use identical patches and grids")
        groups = []
        for view in result["views"]:
            disparity = cache["js_" + view]
            for label in range(6):
                for index in range(len(labels)):
                    current, old = labels[index] == label, previous["native_label"][index] == label
                    union = (current | old).sum()
                    if not union:
                        continue
                    stability = (current & old).sum() / union
                    selected = current & eligible[index]
                    for lo, hi in ((0., .5), (.5, .8), (.8, .95), (.95, 1.00001)):
                        group = selected & (cache["confidence"][index] >= lo) & (cache["confidence"][index] < hi)
                        if group.sum() < 20:
                            continue
                        divergence = disparity[index][group]
                        mistakes = labels[index][group] != truth[index][group]
                        split = float(np.median(divergence))
                        high = divergence > split
                        groups.append({"view": view, "image": str(cache["names"][index]),
                            "class": label, "temporal_class_iou": float(stability),
                            "confidence_bin": [lo, min(hi, 1.)], "pixels": int(group.sum()),
                            "high_js_risk": float(mistakes[high].mean()) if high.any() else None,
                            "low_js_risk": float(mistakes[~high].mean()) if (~high).any() else None})
        result["temporal_diagnostic"] = {
            "definition": "image/class IoU between named frozen checkpoints; not pixel posterior",
            "groups": groups}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    rows = ["selector,lambda,coverage,micro_risk,macro_risk,selected,wrong"]
    for item in result["confidence"]:
        rows.append(f"confidence,0,{item['coverage']},{item['micro_risk']},{item['macro_predicted_class_risk']},{item['selected']},{item['wrong']}")
    for view, item in result["views"].items():
        for penalty, points in item["lambda_sweep"].items():
            for point in points:
                rows.append(f"{view},{penalty},{point['coverage']},{point['micro_risk']},{point['macro_predicted_class_risk']},{point['selected']},{point['wrong']}")
    (output / "risk_coverage.csv").write_text("\n".join(rows) + "\n")
    lines = ["# Frozen-teacher reliability diagnostic", "",
             f"Valid labeled pixels: {result['valid_labeled_pixels']:,}. Native label error: {100*result['native_error']:.3f}%.",
             "", "Every selector uses the same native candidate labels and per-predicted-class coverage.",
             "These target labels are research development/evaluation labels, never adaptation supervision.",
             "", "| Selector, lambda=1 | Risk at q=.5 | Macro class risk |",
             "|---|---:|---:|"]
    point = result["confidence"][2]
    lines.append(f"| Confidence | {100*point['micro_risk']:.3f}% | {100*point['macro_predicted_class_risk']:.3f}% |")
    for view, item in result["views"].items():
        if "1.0" in item["lambda_sweep"]:
            point = item["lambda_sweep"]["1.0"][2]
            lines.append(f"| {view} | {100*point['micro_risk']:.3f}% | {100*point['macro_predicted_class_risk']:.3f}% |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str(output / "REPORT.md"), "native_error": result["native_error"]}), flush=True)
    return result


def infer(args):
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("architecture") != "segformer_mit_b2_nvidia_mlp768":
        raise ValueError("This diagnostic expects the locked B2 architecture")
    encoder, classifier = build_segformer_b2(6)
    encoder.load_state_dict(state["feature_extractor"], strict=True)
    classifier.load_state_dict(state["classifier"], strict=True)
    encoder.to(device).eval()
    classifier.to(device).eval()
    reference_encoder, reference_classifier = encoder, classifier
    reference_state, reference_hash = None, None
    if args.reference_checkpoint:
        reference_state = torch.load(args.reference_checkpoint, map_location='cpu', weights_only=False)
        reference_hash = digest(args.reference_checkpoint)
        if (reference_state.get('stage') != 'source' or
                state.get('source_checkpoint_sha256') != reference_hash):
            raise ValueError('Reference must be the frozen parent source checkpoint')
        reference_encoder, reference_classifier = build_segformer_b2(6)
        reference_encoder.load_state_dict(reference_state['feature_extractor'], strict=True)
        reference_classifier.load_state_dict(reference_state['classifier'], strict=True)
        reference_encoder.to(device).eval(); reference_classifier.to(device).eval()
    cache_root = args.data_root / "gsd_views"
    manifest = json.loads((cache_root / "view_manifest.json").read_text())
    names = manifest["diagnostic_names"][:args.limit]
    if args.limit < len(manifest["diagnostic_names"]):
        from scripts.prepare_gsd_views import choose_diagnostic
        selected = choose_diagnostic([{"image": n} for n in manifest["diagnostic_names"]], args.limit)
        names = sorted(selected)
    if not names:
        raise ValueError("No declared diagnostic images")
    collected = {"names": np.asarray(names)}
    arrays = {key: [] for key in ("native_label", "ground_truth", "eligible", "confidence")}
    for view in args.views:
        for prefix in ("js_", "aligned_", "fusion_"):
            arrays[prefix + view] = []
    started = time.perf_counter()
    with torch.no_grad():
        for index, name in enumerate(names):
            record = manifest["records"][name]
            if record["split"] != "target_eval" or not record["label"]:
                raise ValueError("GT may only be read from the declared diagnostic evaluation role")
            native = classifier(encoder(tensor(args.data_root / record["image"]).to(device))[1], (512, 512)).softmax(1)
            confidence, label = native.max(1)
            with Image.open(args.data_root / record["label"]) as image:
                truth = np.asarray(image).copy()
            with Image.open(cache_root / record["common_native_valid"]) as image:
                eligible = np.asarray(image).copy() > 0
            arrays["native_label"].append(label[0].cpu().numpy().astype(np.uint8))
            arrays["confidence"].append(confidence[0].cpu().numpy())
            arrays["ground_truth"].append(truth.astype(np.uint8))
            for view in args.views:
                item = record["views"][view]
                prediction = reference_classifier(reference_encoder(tensor(cache_root / item["image"]).to(device))[1], (512, 512))
                with Image.open(cache_root / item["valid"]) as image:
                    valid = torch.from_numpy(np.asarray(image).copy() > 0)[None].to(device)
                aligned, support = warp_to_native(prediction, [item["geometry"]], valid, normalize=False)
                aligned = aligned.softmax(1)
                eligible &= support[0].cpu().numpy()
                arrays["js_" + view].append(normalized_js(native, aligned)[0].cpu().numpy())
                arrays["aligned_" + view].append(aligned.argmax(1)[0].cpu().numpy().astype(np.uint8))
                arrays["fusion_" + view].append((native + aligned).argmax(1)[0].cpu().numpy().astype(np.uint8))
            arrays["eligible"].append(eligible)
            if (index + 1) % 16 == 0 or index + 1 == len(names):
                print(f"Frozen inference {index+1}/{len(names)}", flush=True)
    collected.update({key: np.stack(values) for key, values in arrays.items()})
    path = args.output / "scores.npz"
    (np.savez if args.uncompressed else np.savez_compressed)(path, **collected)
    metadata = {"checkpoint_sha256": digest(args.checkpoint), "stage": state.get("stage"),
                "iteration": state.get("iteration"), "source_checkpoint_sha256": state.get("source_checkpoint_sha256"),
                "original_checkpoint_sha256": state.get("original_checkpoint_sha256"),
                "bands": manifest["bands"], "view_manifest_sha256": digest(cache_root / "view_manifest.json"),
                "diagnostic_names": names, "views": args.views, "device": str(device),
                "torch": str(torch.__version__), "wall_seconds": time.perf_counter() - started,
                "teacher_image_forwards": len(names) * (1 + len(args.views)),
                "peak_vram_mb": torch.cuda.max_memory_allocated() / 1048576 if device.type == "cuda" else None,
                "target_train_gt_opened": 0, "target_eval_gt_opened": len(names)}
    metadata.update(reference_checkpoint_sha256=reference_hash,
                    view_teacher='frozen_source' if reference_hash else 'native_checkpoint')
    metadata["alignment"] = "continuous logits resampling followed by softmax; matches SND native order"
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    del encoder, classifier, state, reference_encoder, reference_classifier, reference_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not args.infer_only:
        analyze(path, args.output, lambdas=tuple(args.lambdas), previous_cache=args.previous_cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-only", type=Path)
    parser.add_argument("--previous-cache", type=Path)
    parser.add_argument('--reference-checkpoint', type=Path)
    parser.add_argument('--infer-only', action='store_true')
    parser.add_argument('--uncompressed', action='store_true')
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0., .5, 1., 2.])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--views", nargs="+", default=["ratio_1p8", "ratio_2"])
    args = parser.parse_args()
    if args.cache_only:
        analyze(args.cache_only, args.output, lambdas=tuple(args.lambdas), previous_cache=args.previous_cache)
    else:
        if args.checkpoint is None or args.data_root is None or args.limit < 1:
            parser.error("Inference requires checkpoint/data-root and a positive limit")
        infer(args)


if __name__ == "__main__":
    main()
