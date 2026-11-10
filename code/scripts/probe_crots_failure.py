"""Bounded post-hoc CROTS failure probe: inference and cache inspection only.

No optimization, model selection, or modification of the original run.
Target validation GT is used ONLY for this diagnostic, never adaptation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch

from scripts.diagnose_crots_remote import EvaluationImages, load_model, summarize_matrix
from train_SRMP_remote import CLASS_NAMES, TASKS, ManifestDataset, load_checkpoint, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def patch_inventory(directory, classes, size, grid):
    entries = json.loads((directory/"patches.json").read_text())
    prepared = json.loads((directory/"complete.json").read_text())
    if len(entries) != prepared["patch_count"] or len(set(map(tuple, entries))) != len(entries):
        raise ValueError("Patch index count/uniqueness mismatch")
    by_image = {}
    for i, y, x in entries:
        if not 0 <= i < prepared["images"] or not (0 <= y < grid and 0 <= x < grid):
            raise ValueError("Invalid patch coordinates")
        by_image.setdefault(i, []).append((y, x))
    counts = np.zeros(classes, np.int64)
    full_counts = np.zeros(classes, np.int64)
    rare_fractions = []
    side = size//grid
    for i in range(prepared["images"]):
        with Image.open(directory/f"{i:06d}.png") as im:
            label = np.array(im)
        if label.shape != (size, size) or label.max() >= classes:
            raise ValueError("Invalid cached pseudo mask")
        full_counts += np.bincount(label.ravel(), minlength=classes)
        for y, x in by_image.get(i, []):
            patch = label[y*side:(y+1)*side, x*side:(x+1)*side]
            counts += np.bincount(patch.ravel(), minlength=classes)
            rare_fractions.append(float(np.isin(patch, prepared["rare_classes"]).mean()))
    if full_counts.tolist() != prepared["pixel_counts"]:
        raise ValueError("Actual cache differs from recorded pseudo counts")
    fractions = np.array(rare_fractions)
    return {"class_order": list(CLASS_NAMES[classes]), "patch_count": len(entries),
            "rare_classes": prepared["rare_classes"], "pseudo_counts_verified": True,
            "full_pseudo_fraction": (full_counts/full_counts.sum()).tolist(),
            "bank_pixel_fraction": (counts/counts.sum()).tolist(),
            "bank_pixel_counts": counts.tolist(),
            "rare_pixel_fraction_quantiles": dict(zip(["min", "q25", "median", "q75", "max"],
                np.quantile(fractions, [0, .25, .5, .75, 1]).tolist())),
            "patch_fraction_with_under_1pct_rare_pixels": float((fractions < .01).mean()),
            "patch_fraction_with_under_5pct_rare_pixels": float((fractions < .05).mean())}, by_image


@torch.inference_mode()
def confidence_evaluation(model, dataset, classes, device, threshold):
    matrix = np.zeros((classes, classes), np.int64)
    correct_total = raw_kept = prob_kept = raw_correct = prob_correct = count = 0
    probability_sum = 0.0
    for image, label in dataset:
        logits = model(image[None].to(device), size=label.shape)
        raw, pred = logits.max(1)
        probability = logits.softmax(1).max(1).values
        pred, raw, probability = pred[0].cpu(), raw[0].cpu(), probability[0].cpu()
        valid = label != 255
        correct = pred == label
        raw_mask, prob_mask = (raw >= threshold) & valid, (probability >= threshold) & valid
        count += int(valid.sum())
        correct_total += int((correct & valid).sum())
        raw_kept += int(raw_mask.sum()); prob_kept += int(prob_mask.sum())
        raw_correct += int((correct & raw_mask).sum()); prob_correct += int((correct & prob_mask).sum())
        probability_sum += float(probability[valid].double().sum())
        matrix += torch.bincount(label[valid]*classes+pred[valid], minlength=classes**2).reshape(classes, classes).numpy()
    result = summarize_matrix(matrix)
    result.update(image_count=len(dataset), manifest_row_indices=dataset.indices,
                  pixel_accuracy=correct_total/count, mean_max_softmax=probability_sum/count,
                  raw_threshold_fraction=raw_kept/count, softmax_threshold_fraction=prob_kept/count,
                  raw_threshold_accuracy=raw_correct/raw_kept if raw_kept else None,
                  softmax_threshold_accuracy=prob_correct/prob_kept if prob_kept else None)
    return result


def main():
    if "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        raise RuntimeError("Use one Slurm task")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT/"data/remote_sensing/dgle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("This bounded probe requires a positive sample size")
    protocol = json.loads((args.run/"protocol.json").read_text())
    config = protocol["config"]
    task, classes = config["task"], config["num_classes"]
    if protocol.get("smoke_only"):
        raise ValueError("Expected a real completed run")
    if sha256_file(args.source_checkpoint) != protocol["source_checkpoint_sha256"]:
        raise ValueError("Wrong source checkpoint")
    for name, expected in protocol["code_sha256"].items():
        actual = hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError("Code changed since original run: "+name)
    for split, key in ((TASKS[task][2], "target_manifest_sha256"), (TASKS[task][3], "evaluation_manifest_sha256")):
        if sha256_file(args.data_root/f"manifests/{split}.csv") != protocol[key]:
            raise ValueError("Manifest changed")
    prepared = json.loads((args.run/"pseudo/complete.json").read_text())
    if prepared["protocol_sha256"] != protocol["signature"]:
        raise ValueError("Cache protocol mismatch")
    last = args.run/f"adapt_round{config['rounds']}/last.pth"
    result = json.loads((args.run/"result_portable.json").read_text())
    digest = sha256_file(last)
    if digest != result["evaluated_checkpoint_sha256"]["crots"]:
        raise ValueError("Checkpoint changed since final evaluation")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"diagnostic_only": True, "contains_weights": False, "optimization_steps": 0,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "task": task,
              "scope": "post-hoc native-resolution target-val sample; no checkpoint selection",
              "source_checkpoint_sha256": protocol["source_checkpoint_sha256"],
              "adapt_checkpoint_sha256": digest, "original_protocol_signature": protocol["signature"],
              "code_dependencies_verified": True, "probe_sha256": sha256_file(Path(__file__)),
              "versions": {"torch": torch.__version__, "numpy": np.__version__}, "precision": "FP32"}
    def save():
        path = args.output/"probe.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        with zipfile.ZipFile(args.output/"probe_portable.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(path, "probe.json")
    report["patch_bank"], by_image = patch_inventory(args.run/"pseudo", classes, config["image_size"], config["patch_grid"])
    save()
    print("Bank pixel fraction:", report["patch_bank"]["bank_pixel_fraction"], flush=True)
    dataset = EvaluationImages(args.data_root, TASKS[task][3], classes, 0, args.limit)
    source = load_model(args.source_checkpoint, task, device)
    report["source"] = confidence_evaluation(source, dataset, classes, device, config["confidence_threshold"])
    print("Source:", report["source"]["miou"], flush=True)
    source_buffers = {k: v.cpu().clone() for k, v in source.named_buffers()}
    # Check patch labels against the SAME source model evaluated on isolated
    # crops. These labels are pseudo labels, not ground truth.
    import csv
    with (args.data_root/f"manifests/{TASKS[task][2]}.csv").open(newline="") as stream:
        train_rows = list(csv.DictReader(stream))
    ids = sorted(by_image)
    chosen = [ids[i] for i in np.linspace(0, len(ids)-1, min(args.limit, len(ids)), dtype=int)]
    agreement = np.zeros((classes, classes), np.int64)
    cache_mismatches = pixels = 0
    with torch.inference_mode():
        for i in chosen:
            with Image.open(args.data_root/train_rows[i]["image"]) as im:
                im = im.convert("RGB").resize((config["image_size"],)*2, Image.Resampling.BICUBIC)
                x = ManifestDataset._tensor(im)[None].to(device)
            with Image.open(args.run/f"pseudo/{i:06d}.png") as im:
                pseudo = torch.from_numpy(np.array(im, dtype=np.int64))
            logits = source(x, size=x.shape[-2:]) + source(x.flip(-1), size=x.shape[-2:]).flip(-1)
            full_pred = logits.argmax(1)[0].cpu()
            cache_mismatches += int((full_pred != pseudo).sum()); pixels += pseudo.numel()
            side = config["image_size"]//config["patch_grid"]
            for y, col in by_image[i]:
                patch = x[:, :, y*side:(y+1)*side, col*side:(col+1)*side]
                labels = pseudo[y*side:(y+1)*side, col*side:(col+1)*side]
                pred = source(patch, size=labels.shape).argmax(1)[0].cpu()
                agreement += torch.bincount((labels*classes+pred).ravel(), minlength=classes**2).reshape(classes, classes).numpy()
    totals = agreement.sum(1)
    report["source_patch_context"] = {"target_train_image_indices": chosen,
        "full_flip_cache_disagreement_fraction": cache_mismatches/pixels,
        "comparison": "cached full-image pseudo label versus isolated-patch source argmax; NOT GT accuracy",
        "class_order": list(CLASS_NAMES[classes]), "cached_pseudo_rows_patch_prediction_columns": agreement.tolist(),
        "per_class_pseudo_agreement": [float(agreement[i,i]/totals[i]) if totals[i] else None for i in range(classes)]}
    del source
    save()
    model = load_model(last, task, device, protocol["signature"])
    report["student"] = confidence_evaluation(model, dataset, classes, device, config["confidence_threshold"])
    report["frozen_buffers_unchanged_student"] = all(torch.equal(source_buffers[k], v.cpu()) for k,v in model.named_buffers())
    print("Student:", report["student"]["miou"], flush=True)
    save()
    state = load_checkpoint(last)
    if state["completed_epochs"] != config["adapt_epochs_per_round"]:
        raise ValueError("Incomplete final stage")
    model.load_state_dict(state["teacher"], strict=True)
    del state
    report["ema_teacher"] = confidence_evaluation(model, dataset, classes, device, config["confidence_threshold"])
    report["frozen_buffers_unchanged_teacher"] = all(torch.equal(source_buffers[k], v.cpu()) for k,v in model.named_buffers())
    print("EMA teacher:", report["ema_teacher"]["miou"], flush=True)
    save()
    print(args.output/"probe_portable.zip", flush=True)


if __name__ == "__main__":
    main()
