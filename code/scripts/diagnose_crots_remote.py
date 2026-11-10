"""Post-hoc CROTS scale/source diagnostics. Does not train or select a model.

Run as `python -m scripts.diagnose_crots_remote`. Ground truth stays at its
original resolution even when the input image is resized. Reports contain no
weights and never replace the original experiment's result files.
"""
from __future__ import annotations

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
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_SRMP_remote import (CLASS_NAMES, TASKS, ManifestDataset, load_checkpoint,
                               make_model, sha256_file)

ROOT = Path(__file__).resolve().parents[1]


class EvaluationImages(Dataset):
    def __init__(self, root, split, classes, input_size, limit=0):
        # Eval role also permits an explicit SOURCE-domain diagnostic. It reads
        # native images/masks deterministically, without training augmentation.
        self.base = ManifestDataset(root, split, "eval", classes, 512)
        self.indices = list(range(len(self.base))) if not limit else sorted(set(
            np.linspace(0, len(self.base)-1, min(limit, len(self.base)), dtype=int).tolist()))
        self.input_size = input_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image_path, label_path = self.base.rows[self.indices[index]]
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
            if self.input_size:
                image = image.resize((self.input_size, self.input_size), Image.Resampling.BICUBIC)
            tensor = ManifestDataset._tensor(image)
        with Image.open(label_path) as loaded:
            label = torch.from_numpy(np.array(loaded, dtype=np.int64, copy=True))
        if label.ndim != 2 or torch.any((label >= self.base.classes) & (label != 255)):
            raise ValueError("Invalid evaluation mask")
        return tensor, label


class Model(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.encoder, self.classifier = make_model(classes)

    def forward(self, image, size):
        # Resize floating logits to native GT before argmax; never resize GT.
        return self.classifier(self.encoder(image)[1], size)


def load_model(path, task, device, signature=None):
    state = load_checkpoint(path)
    model = Model(TASKS[task][0])
    if "student" in state:
        if not signature or state["protocol_signature"] != signature:
            raise ValueError("Student checkpoint protocol mismatch")
        model.load_state_dict(state["student"], strict=True)
    else:
        expected = {"task": task, "stage": "source", "num_classes": TASKS[task][0],
                    "source_split": TASKS[task][1], "normalization": "imagenet_rgb_planes"}
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"Source mismatch: {key}")
        if state.get("iteration") != 20000:
            raise ValueError("Expected a completed 20000-step source checkpoint")
        model.encoder.load_state_dict(state["feature_extractor"], strict=True)
        model.classifier.load_state_dict(state["classifier"], strict=True)
    return model.to(device).eval()


def summarize_matrix(matrix, names=None):
    matrix = np.asarray(matrix, dtype=np.int64)
    predicted, gt, intersection = matrix.sum(0), matrix.sum(1), matrix.diagonal()
    union = predicted + gt - intersection
    def ratio(numerator, denominator):
        return np.divide(numerator, denominator, out=np.full(len(union), np.nan), where=denominator > 0)
    iou = ratio(intersection, union)
    accuracy = ratio(intersection, gt)
    precision = ratio(intersection, predicted)
    f1 = ratio(2 * intersection, predicted + gt)
    names = CLASS_NAMES[len(union)] if names is None else names
    if len(names) != len(union):
        raise ValueError('Class names and confusion matrix dimensions differ')
    result = {"class_order": list(names), "confusion_matrix": matrix.tolist(),
              "confusion_axes": "rows=ground_truth, columns=prediction",
              "iou": [None if np.isnan(v) else float(v) for v in iou],
              "iou_by_class": {k: None if np.isnan(v) else float(v) for k, v in zip(names, iou)},
              "miou": float(np.nanmean(iou)), "valid_pixels": int(matrix.sum()),
              "gt_fraction": (matrix.sum(1)/max(int(matrix.sum()), 1)).tolist(),
              "prediction_fraction": (matrix.sum(0)/max(int(matrix.sum()), 1)).tolist()}
    for key, values in (("accuracy", accuracy), ("precision", precision), ("recall", accuracy), ("f1", f1)):
        result[key + "_by_class"] = {name: float(value) if np.isfinite(value) else None
                                     for name, value in zip(names, values)}
    result.update(overall_accuracy=float(intersection.sum() / max(int(gt.sum()), 1)),
                  mean_accuracy=float(np.nanmean(accuracy)), mean_f1=float(np.nanmean(f1)),
                  mean_precision=float(np.nanmean(precision)), mean_recall=float(np.nanmean(accuracy)))
    if len(union) == 6:
        result["miou5"] = float(np.nanmean(iou[:5]))
    return result


@torch.inference_mode()
def evaluate_at_size(model, dataset, classes, device, workers, amp):
    matrix = torch.zeros((classes, classes), dtype=torch.int64)
    for images, labels in DataLoader(dataset, batch_size=1, num_workers=workers, shuffle=False):
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(images.to(device), size=labels.shape[-2:])
        prediction = logits.argmax(1).cpu()
        valid = labels != 255
        encoded = labels[valid]*classes + prediction[valid]
        matrix += torch.bincount(encoded, minlength=classes**2).reshape_as(matrix)
    return summarize_matrix(matrix.numpy())


def main():
    if "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        raise RuntimeError("This diagnostic is single-process; use srun --ntasks=1 to avoid duplicate writers")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="r2u")
    parser.add_argument("--run", type=Path, help="Existing CROTS run; omit for source-only diagnosis")
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--source-domain", action="store_true", help="Source TRAIN images: fit diagnostic, not held-out accuracy")
    parser.add_argument("--data-root", type=Path, default=ROOT/"data/remote_sensing/dgle")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0=all; otherwise evenly spaced manifest sample")
    parser.add_argument("--sizes", type=int, nargs="+", default=[0, 512], help="0=native input; GT is always native")
    parser.add_argument("--models", nargs="+", help="Optional checkpoint names, e.g. source distill round3; default=all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or any(n < 0 for n in args.sizes):
        parser.error("Sizes and limit must be nonnegative")
    if args.source_domain and args.run:
        parser.error("Source-domain diagnostic is separate from adaptation; omit --run")
    classes, source_split, _, target_split = TASKS[args.task]
    source = args.source_checkpoint or ROOT/f"results_remote_sensing/{args.task}/rs_standard/crop512_batch4_accum1_source20000/seed{args.seed}/source/source_last.pth"
    split = source_split if args.source_domain else target_split
    output = args.output or ROOT/f"results_remote_sensing/{args.task}/diagnostics/seed{args.seed}/{split}_n{args.limit}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = {"source": source}
    protocol = None
    recorded = None
    if args.run:
        protocol = json.loads((args.run/"protocol.json").read_text())
        if protocol["config"]["task"] != args.task or protocol.get("smoke_only"):
            raise ValueError("Run task mismatch or smoke run")
        if sha256_file(source) != protocol["source_checkpoint_sha256"]:
            raise ValueError("Different source checkpoint from recorded CROTS run")
        if sha256_file(args.data_root/f"manifests/{target_split}.csv") != protocol["evaluation_manifest_sha256"]:
            raise ValueError("Evaluation manifest changed")
        if sha256_file(args.data_root/"metadata.json") != protocol["dataset_metadata_sha256"]:
            raise ValueError("Dataset metadata changed")
        if (args.run/"distill/last.pth").exists():
            checkpoints["distill"] = args.run/"distill/last.pth"
        for i in range(1, protocol["config"]["rounds"]+1):
            checkpoints[f"round{i}"] = args.run/f"adapt_round{i}/last.pth"
        if (args.run/"result_portable.json").exists():
            recorded = json.loads((args.run/"result_portable.json").read_text())
        # Architecture/evaluator dependencies must match the completed run.
        for name, digest in protocol["code_sha256"].items():
            if name == "train_SRMP_remote.py" or name.startswith("core/models/"):
                actual = hashlib.sha256((ROOT/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                if actual != digest:
                    raise ValueError("Model dependency changed since training: "+name)
    if args.models:
        if any(name not in checkpoints for name in args.models):
            raise ValueError("Unknown checkpoint requested; available: "+", ".join(checkpoints))
        checkpoints = {name: checkpoints[name] for name in dict.fromkeys(args.models)}
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Activate the training environment on your allocated GPU node")
    torch.set_num_threads(4)
    report = {"diagnostic_only": True, "contains_weights": False, "task": args.task,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "split": split,
              "source_training_set_diagnostic": args.source_domain, "limit": args.limit,
              "checkpoint_selection": "none; every requested checkpoint reported",
              "ground_truth_policy": "native labels; resize logits before argmax; ignore=255",
              "input_resize": "PIL bicubic RGB before normalization", "amp": not args.no_amp and device.type == "cuda",
              "manifest_sha256": sha256_file(args.data_root/f"manifests/{split}.csv"),
              "code_sha256": sha256_file(Path(__file__)), "versions": {"torch": torch.__version__, "numpy": np.__version__},
              "metrics": {}, "checkpoint_sha256": {}}
    for name, checkpoint in checkpoints.items():
        digest = sha256_file(checkpoint)
        original_name = name
        if protocol and name == f"round{protocol['config']['rounds']}":
            original_name = "crots"
        if recorded and original_name in recorded["evaluated_checkpoint_sha256"]:
            if digest != recorded["evaluated_checkpoint_sha256"][original_name]:
                raise ValueError("Checkpoint changed after original evaluation: "+name)
        model = load_model(checkpoint, args.task, device, protocol["signature"] if protocol else None)
        report["checkpoint_sha256"][name] = digest
        report["metrics"][name] = {}
        for size in dict.fromkeys(args.sizes):
            dataset = EvaluationImages(args.data_root, split, classes, size, args.limit)
            key = "native" if size == 0 else str(size)
            metrics = evaluate_at_size(model, dataset, classes, device, args.workers, not args.no_amp)
            metrics["image_count"] = len(dataset)
            metrics["manifest_row_indices"] = dataset.indices
            report["metrics"][name][key] = metrics
            print(f"{name} input={key} images={len(dataset)} mIoU={100*metrics['miou']:.4f}%", flush=True)
        del model
        # Persist completed models if a later evaluation is interrupted.
        (output/"diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        with zipfile.ZipFile(output/"diagnostic_portable.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(output/"diagnostic.json", "diagnostic.json")
    print(output/"diagnostic_portable.zip")


if __name__ == "__main__":
    main()
