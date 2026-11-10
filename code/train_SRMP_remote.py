"""Source training and source-free SRMP for the two DGLE remote-sensing tasks.

Reads the prepared CSV manifests directly. Target training rows must have no
label path; only the separate evaluation command opens target ground truth.
The SRMP relation, masking, rebuilder, and frozen ASPP geometry are shared with
train_SRMP.py. Cityscapes-specific candidate/copy-paste code is not used.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
import random
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageEnhance, __version__ as pillow_version
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_SRMP import (
    SpatialRebuilder,
    auxiliary_weight,
    masked_mean,
    relational_cosine_map,
    sample_block_mask,
    source_kernel_geometry,
    update_ema,
)
from scripts.remote_sensing_transforms import photometric, siamseg_photometric, train_transform
from scripts.remote_sensing_results import write_portable_result


TASKS = {
    "vh2pd": (6, "isprs_vh_irrg_train", "isprs_pd_rgb_train", "isprs_pd_rgb_eval"),
    "r2u": (7, "loveda_rural_train", "loveda_urban_train", "loveda_urban_val"),
}
B2_SPEC = (6, "source_train", "target_train", "target_eval")
B2_PROTOCOL = "siamseg_sfuda_b2_v1"
CLASS_NAMES = {
    6: ("impervious_surface", "building", "low_vegetation", "tree", "car", "clutter"),
    7: ("background", "building", "road", "water", "barren", "forest", "agriculture"),
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
LOGGER = logging.getLogger("srmp.remote")


class ManifestDataset(Dataset):
    def __init__(self, root: Path, split: str, role: str, classes: int, crop: int,
                 policy: str = "rs_standard"):
        if role not in ("source", "target", "eval"):
            raise ValueError(role)
        self.root, self.role, self.classes, self.crop = root.resolve(), role, classes, crop
        self.policy = policy
        manifest = self.root / "manifests" / f"{split}.csv"
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or set(rows[0]) != {"image", "label"}:
            raise ValueError(f"Empty or malformed manifest: {manifest}")
        self.rows = []
        for row in rows:
            image = self._path(row["image"])
            label = self._path(row["label"]) if row["label"] else None
            if not image.is_file() or (label is not None and not label.is_file()):
                raise FileNotFoundError(f"Missing file in {manifest}: {row}")
            if (role == "target") != (label is None):
                raise ValueError(f"Incorrect label visibility for {split}: {row}")
            self.rows.append((image, label))

    def _path(self, value: str) -> Path:
        path = (self.root / value).resolve()
        if os.path.commonpath((str(self.root), str(path))) != str(self.root):
            raise ValueError(f"Manifest path escapes dataset root: {value}")
        return path

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _tensor(image: Image.Image) -> torch.Tensor:
        # Vaihingen's stored RGB planes contain NIR, red, green, in that order.
        array = np.asarray(image, dtype=np.uint8).copy().transpose(2, 0, 1)
        return torch.from_numpy((array.astype(np.float32) / 255.0 - MEAN) / STD)

    def __getitem__(self, index: int):
        image_path, label_path = self.rows[index]
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        label = None
        if label_path is not None:
            with Image.open(label_path) as loaded:
                label = loaded.convert("L")
            if label.size != image.size:
                raise ValueError(f"Image/label size mismatch: {image_path}")
        if self.role != "eval":
            image, label, valid = train_transform(image, label, self.crop, self.policy)
        clean = self._tensor(image)
        if self.role == "target":
            # Photometric change only; teacher/student pixels remain aligned.
            if self.policy == "rs_standard":
                augmented = photometric(image)
            else:
                augmented = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
                augmented = ImageEnhance.Contrast(augmented).enhance(random.uniform(0.8, 1.2))
            valid_tensor = torch.from_numpy(np.asarray(valid, dtype=np.uint8).copy() != 0)
            return clean, self._tensor(augmented), valid_tensor
        if self.role == "source" and self.policy == "rs_standard":
            clean = self._tensor(photometric(image))
        elif self.role == "source" and self.policy in ("siamseg_isprs", "siamseg_sfuda"):
            clean = self._tensor(siamseg_photometric(image))
        mask = torch.from_numpy(np.asarray(label, dtype=np.uint8).astype(np.int64))
        if torch.any((mask >= self.classes) & (mask != 255)):
            raise ValueError(f"Unexpected class ID in {label_path}")
        return clean, mask


def make_model(classes: int, freeze_bn: bool = True, architecture: str = "resnet101_aspp"):
    if architecture == "segformer_mit_b2_nvidia_mlp768":
        from core.models.segformer import build_segformer_b2
        return build_segformer_b2(classes)
    from core.models.classifier import ASPP_Classifier_V2
    from core.models.feature_extractor import resnet_feature_extractor

    encoder = resnet_feature_extractor(
        "resnet101", aux=True, pretrained_backbone=False, freeze_bn=freeze_bn
    )
    classifier = ASPP_Classifier_V2(2048, [6, 12, 18, 24], [6, 12, 18, 24], classes)
    return encoder, classifier


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.x
        return torch.load(path, map_location="cpu")


def initialize_backbone(encoder: nn.Module, path: Path) -> None:
    checkpoint = load_checkpoint(path)
    if encoder.__class__.__name__ == 'MiTB2Encoder':
        from core.models.segformer import load_imagenet_b2
        load_imagenet_b2(encoder, checkpoint)
        return
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError(f"Invalid ImageNet backbone checkpoint: {path}")
    normalized = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("backbone."):
            key = key[9:]
        normalized[key] = value
    state = normalized
    available = encoder.backbone.state_dict()
    usable = {key: value for key, value in state.items()
              if key in available and available[key].shape == value.shape}
    missing = sorted(set(available) - set(usable))
    if missing:
        raise ValueError(f"ImageNet backbone checkpoint is incomplete/incompatible: {path}; "
                         f"missing {len(missing)} tensors, e.g. {missing[:5]}")
    encoder.backbone.load_state_dict(usable, strict=True)
    LOGGER.info("Loaded %d ResNet tensors from %s", len(usable), path)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def loader(dataset: Dataset, args, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed + (0 if dataset.role == "source" else 1))
    return DataLoader(dataset, batch_size=args.batch_size if shuffle else 1,
                      shuffle=shuffle, num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                      drop_last=shuffle, persistent_workers=args.workers > 0,
                      worker_init_fn=seed_worker, generator=generator)


def infinite_batches(data_loader):
    while True:
        yield from data_loader


def evaluate(encoder, classifier, dataset, args, device) -> dict:
    encoder.eval()
    classifier.eval()
    matrix = torch.zeros((dataset.classes, dataset.classes), dtype=torch.int64)
    with torch.no_grad():
        for index, (images, labels) in enumerate(loader(dataset, args, shuffle=False)):
            if getattr(args, 'eval_limit', 0) and index >= args.eval_limit:
                break
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = classifier(encoder(images)[1], images.shape[-2:])
            prediction = logits.argmax(1).cpu()
            valid = labels != 255
            encoded = labels[valid] * dataset.classes + prediction[valid]
            matrix += torch.bincount(encoded, minlength=dataset.classes ** 2).reshape_as(matrix)
    tp = matrix.diag().double()
    union = matrix.sum(0).double() + matrix.sum(1).double() - tp
    iou = torch.where(union > 0, tp / union.clamp_min(1), torch.nan)
    result = {"iou": [None if math.isnan(x) else float(x) for x in iou.tolist()],
              "miou": float(torch.nanmean(iou)), "valid_pixels": int(matrix.sum())}
    result["iou_by_class"] = dict(zip(CLASS_NAMES[dataset.classes], result["iou"]))
    result["confusion_matrix"] = matrix.tolist()  # rows: GT, columns: prediction
    precision = tp / matrix.sum(0).double().clamp_min(1)
    recall = tp / matrix.sum(1).double().clamp_min(1)
    result["macro_f1"] = float(torch.nanmean(torch.where(matrix.sum(1) > 0,
        2 * precision * recall / (precision + recall).clamp_min(1e-12), torch.nan)))
    result["overall_accuracy"] = float(tp.sum() / matrix.sum().clamp_min(1))
    if dataset.classes == 6:
        result["miou5"] = float(torch.nanmean(iou[:5]))
    return result


def write_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_result(args, metrics: dict, checkpoint: dict, evaluated_path: Path,
                  training_protocol: dict | None = None) -> None:
    """Export scalar metadata and metrics only; a report needs no checkpoint to read."""
    evaluated_hash = sha256_file(evaluated_path)
    stage = checkpoint.get("stage", "unknown")
    source_hash = (checkpoint.get("source_checkpoint_sha256")
                   if stage == "adapt" else evaluated_hash)
    model = {key: checkpoint[key] for key in (
        "stage", "iteration", "num_classes", "seed", "train_policy",
        "normalization", "crop_size", "training_settings", "architecture",
        "protocol", "protocol_sha256", "target_eval_manifest_sha256",
        "checkpoint_selection", "smoke_only") if key in checkpoint}
    model.update(source_checkpoint_sha256=source_hash,
                 evaluated_checkpoint_sha256=evaluated_hash,
                 inference_model="ema_teacher" if stage == "adapt" else "source_encoder")
    protocol = json.loads((args.output_dir / f"protocol_{args.stage}.json").read_text(encoding="utf-8"))
    if training_protocol is None:
        original_protocol = evaluated_path.parent / f"protocol_{stage}.json"
        if original_protocol.is_file():
            training_protocol = json.loads(original_protocol.read_text(encoding="utf-8"))
    eval_split = (B2_SPEC if args.protocol == B2_PROTOCOL else TASKS[args.task])[3]
    evaluation_protocol = {
        "task": args.task, "stage": "eval", "split": eval_split,
        "inference_model": model["inference_model"],
        "args": {"data_root": str(args.data_root), "amp": args.amp,
                 "workers": args.workers, "evaluation_batch_size": 1},
        "files_sha256": {
            "manifest": sha256_file(args.data_root / "manifests" / f"{eval_split}.csv"),
            "dataset_metadata": sha256_file(args.data_root / "metadata.json"),
            "evaluated_checkpoint": evaluated_hash,
            "trainer": sha256_file(Path(__file__)),
            "transforms": sha256_file(Path(__file__).parent / "scripts/remote_sensing_transforms.py"),
        },
        "versions": protocol["versions"],
        "selection_rule": protocol["selection_rule"],
    }
    baseline_path = args.source_baseline
    if baseline_path is None and args.stage == "adapt":
        baseline_path = args.source_checkpoint.parent / "source_eval" / "metrics_eval.json"
    destination = args.output_dir / "result_portable.json"
    report = write_portable_result(
        destination, task=args.task, split=eval_split, metrics=metrics,
        model=model, protocol=evaluation_protocol, training_protocol=training_protocol,
        dataset_metadata=json.loads((args.data_root / "metadata.json").read_text(encoding="utf-8")),
        source_baseline_path=baseline_path,
    )
    LOGGER.info("Portable result (no weights): %s", destination)
    LOGGER.info("Source comparison: %s", json.dumps(report["source_comparison"]))


def record_protocol(args, spec) -> None:
    """Record the exact input split and setup without opening target labels."""
    split = spec[{"source": 1, "adapt": 2, "eval": 3}[args.stage]]
    files = {
        "manifest": args.data_root / "manifests" / f"{split}.csv",
        "dataset_metadata": args.data_root / "metadata.json",
        "trainer": Path(__file__),
        "transforms": Path(__file__).parent / "scripts" / "remote_sensing_transforms.py",
        "srmp_loss": Path(__file__).parent / "train_SRMP.py",
        "remask_reliability": Path(__file__).parent / "train_ReMask.py",
    }
    if args.protocol == B2_PROTOCOL:
        files['protocol_config'] = Path(__file__).parent / 'configs/remote_sensing/siamseg_sfuda_b2_v1.json'
    if args.stage == "source":
        files["initial_backbone"] = args.backbone_weights
        if args.source_val_split:
            files["source_val_manifest"] = args.data_root / "manifests" / f"{args.source_val_split}.csv"
        if args.eval_period:
            files["target_eval_manifest"] = args.data_root / "manifests" / f"{spec[3]}.csv"
    else:
        files["source_checkpoint"] = args.source_checkpoint
    protocol = {
        "task": args.task, "stage": args.stage, "split": split,
        "args": {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
        "files_sha256": {key: sha256_file(path) for key, path in files.items()},
        "versions": {"torch": torch.__version__, "numpy": np.__version__,
                     "pillow": pillow_version},
        "selection_rule": ("smoke_only_no_evaluation" if args.protocol == B2_PROTOCOL and args.stage == 'source' and args.no_eval else
                           "target_validation_selected; earliest checkpoint breaks mIoU6 ties"
                           if args.protocol == B2_PROTOCOL and args.stage == 'source' else
                           "best source-domain validation mIoU6; target metrics are diagnostic only"
                           if args.stage == "source" and args.source_val_split else
                           "last scheduled checkpoint; target validation never selects weights"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"protocol_{args.stage}.json"
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def source_train(args, spec, device):
    classes, source_split, _, _ = spec
    b2 = args.protocol == B2_PROTOCOL
    protocol = json.loads((args.output_dir / "protocol_source.json").read_text(encoding="utf-8"))
    dataset = ManifestDataset(args.data_root, source_split, "source", classes,
                              args.crop_size, args.train_policy)
    if len(dataset) < args.batch_size:
        raise ValueError("Source dataset is smaller than one batch")
    encoder, classifier = make_model(classes, architecture=('segformer_mit_b2_nvidia_mlp768' if b2 else 'resnet101_aspp'))
    initialize_backbone(encoder, args.backbone_weights)
    encoder, classifier = encoder.to(device), classifier.to(device)
    encoder.train()
    classifier.train()
    if b2:
        groups = []
        for module, part, multiplier in ((encoder, 'backbone', 1.), (classifier, 'decoder', args.head_lr_mult)):
            for decay in (True, False):
                selected = [(name, parameter) for name, parameter in module.named_parameters()
                            if parameter.requires_grad and (parameter.ndim > 1 and not name.endswith('bias')) == decay]
                groups.append({'params': [parameter for _, parameter in selected], 'lr_mult': multiplier,
                               'weight_decay': .01 if decay else 0., 'name': f'{part}_{"decay" if decay else "no_decay"}'})
                LOGGER.info('AdamW group %s: %d parameters, %d tensors, lr multiplier %s', groups[-1]['name'],
                            sum(p.numel() for _, p in selected), len(selected), multiplier)
        optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(.9, .999))
    else:
        optimizer = torch.optim.SGD([
            {"params": encoder.parameters(), "lr_mult": 1.},
            {"params": classifier.parameters(), "lr_mult": args.head_lr_mult},
        ], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    batches = infinite_batches(loader(dataset, args, shuffle=True))
    evaluations = {}
    if args.source_val_split:
        source_val = ManifestDataset(args.data_root, args.source_val_split, "eval", classes, args.crop_size)
        # Tile-level independence, not a random split of overlapping chips.
        if b2:
            setup = json.loads((Path(__file__).parent / 'configs/remote_sensing/siamseg_sfuda_b2_v1.json').read_text())
            train_expected, val_expected = set(setup['source_train_tiles']), set(setup['source_val_tiles'])
        else:
            from scripts.prepare_dgle_remote_sensing import VH_TRAIN, VH_VAL
            train_expected, val_expected = set(VH_TRAIN), set(VH_VAL)
        train_tiles = {int(p.stem.split("_")[0]) for p, _ in dataset.rows}
        val_tiles = {int(p.stem.split("_")[0]) for p, _ in source_val.rows}
        if train_tiles != train_expected or val_tiles != val_expected or train_tiles & val_tiles:
            raise ValueError("Source training/validation must use the fixed disjoint VH tile lists")
        evaluations["source_val"] = source_val
    if args.eval_period:
        evaluations["target_eval"] = ManifestDataset(args.data_root, spec[3], "eval", classes, args.crop_size)
    curve, peaks = [], {}
    if b2 and args.resume_training:
        existing_curve = args.output_dir/'evaluation_curve.jsonl'
        if existing_curve.is_file():
            curve = [json.loads(line) for line in existing_curve.read_text(encoding='utf-8').splitlines()]
        existing_report = args.output_dir/'source_training_report.json'
        if existing_report.is_file():
            peaks = json.loads(existing_report.read_text(encoding='utf-8')).get('observed_peaks', {})

    def payload(step, selection="last_scheduled"):
        return {
            "task": args.task, "num_classes": classes, "stage": "source", "iteration": step,
            "feature_extractor": encoder.state_dict(), "classifier": classifier.state_dict(),
            "crop_size": args.crop_size, "normalization": "imagenet_rgb_planes",
            "train_policy": args.train_policy, "seed": args.seed, "source_split": source_split,
            "source_manifest_sha256": protocol["files_sha256"]["manifest"],
            "backbone_sha256": protocol["files_sha256"]["initial_backbone"],
            "training_settings": {"steps": args.steps, "lr": args.lr,
                                  "head_lr_mult": args.head_lr_mult, "amp": args.amp},
            "checkpoint_selection": selection,
            "source_val_manifest_sha256": protocol["files_sha256"].get("source_val_manifest"),
            **({'architecture': 'segformer_mit_b2_nvidia_mlp768', 'decoder_width': 768,
                'feature_channels': [64, 128, 320, 512], 'protocol': B2_PROTOCOL,
                'protocol_sha256': protocol['files_sha256']['protocol_config'],
                'target_eval_manifest_sha256': protocol['files_sha256'].get('target_eval_manifest'),
                'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'rng': rng_state(),
                'smoke_only': args.smoke_only} if b2 else {}),
        }

    def observe(step):
        if not evaluations:
            return
        state, modes = rng_state(), (encoder.training, classifier.training)
        record = {"iteration": step}
        try:
            # Always FP32 for comparable diagnostic metrics, even under AMP training.
            eval_args = SimpleNamespace(**{**vars(args), "amp": False})
            for name, validation in evaluations.items():
                record[name] = evaluate(encoder, classifier, validation, eval_args, device)
                if step > 0 and record[name]["miou"] > peaks.get(name, {}).get("miou", -1):
                    peaks[name] = {"iteration": step, "miou": record[name]["miou"]}
                    if name == "source_val":
                        write_checkpoint(args.output_dir / ("best_source_val.pth" if b2 else "source_best.pth"), payload(step, "best_source_val_miou6"))
                    elif name == 'target_eval' and b2:
                        write_checkpoint(args.output_dir / 'best_target_val.pth', payload(step, 'target_validation_selected'))
                LOGGER.info("source curve step %d %s mIoU %.4f%%", step, name, record[name]["miou"] * 100)
        finally:
            encoder.train(modes[0]); classifier.train(modes[1])
            restore_rng(state)
        curve.append(record)
        with (args.output_dir / "evaluation_curve.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        selected_step = peaks.get("target_eval" if b2 else "source_val", {}).get("iteration")
        selected = next((row for row in curve if row["iteration"] == selected_step), None)
        report = {"task": args.task, "completed": step == args.steps,
                  "selection_rule": protocol["selection_rule"], "contains_weights": False,
                  "source_best": selected, "last_observation": record, "observed_peaks": peaks,
                  "target_peak_use": "development checkpoint selection" if b2 else "oracle diagnostic; never selects checkpoint or stops training",
                  "evaluation_precision": "float32",
                  "initialization": "ImageNet backbone plus newly initialized SegFormer head" if b2 else "ImageNet backbone plus newly initialized ASPP head",
                  "protocol": protocol}
        if step == args.steps:
            report["checkpoint_sha256"] = {name: sha256_file(args.output_dir / name)
                for name in (("best_source_val.pth", "best_target_val.pth", "last.pth") if b2 else ("source_best.pth", "source_last.pth")) if (args.output_dir / name).exists()}
        (args.output_dir / "source_training_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary = args.output_dir / "results_portable.zip.tmp"
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("source_training_report.json", "evaluation_curve.jsonl", "protocol_source.json"):
                archive.write(args.output_dir / name, name)
        os.replace(temporary, args.output_dir / "results_portable.zip")

    start = 0
    if b2 and args.resume_training:
        resume = load_checkpoint(args.resume_training)
        if (resume.get('architecture') != 'segformer_mit_b2_nvidia_mlp768' or
                resume.get('protocol_sha256') != protocol['files_sha256']['protocol_config'] or
                resume.get('source_manifest_sha256') != protocol['files_sha256']['manifest'] or
                resume.get('target_eval_manifest_sha256') != protocol['files_sha256'].get('target_eval_manifest') or
                resume.get('training_settings', {}).get('steps') != args.steps or
                resume.get('training_settings', {}).get('lr') != args.lr or
                resume.get('smoke_only') != args.smoke_only):
            raise ValueError('Source resume provenance mismatch')
        encoder.load_state_dict(resume['feature_extractor'], strict=True)
        classifier.load_state_dict(resume['classifier'], strict=True)
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        restore_rng(resume['rng'])
        start = resume['iteration']
    if start == 0:
        observe(0)
    for step in range(start + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.accum_steps):
            images, labels = next(batches)
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = classifier(encoder(images)[1], images.shape[-2:])
                ce = F.cross_entropy(logits.float(), labels, ignore_index=255, reduction="none",
                                     weight=(logits.new_tensor([1, 1, 1, 1.25, 1.5, 1.5]) if b2 else None))
                valid = labels != 255
                loss = (ce * valid).sum() / valid.sum().clamp_min(1)
                if b2 and (not torch.isfinite(loss) or not valid.any()):
                    raise FloatingPointError('B2 source batch has no valid pixels or non-finite loss')
            scaler.scale(loss / args.accum_steps).backward()
            total += float(loss.detach()) / args.accum_steps
        lr = (args.lr * (1e-6 + (1 - 1e-6) * step / 1500) if b2 and step <= 1500 else
              args.lr * (1 - (step - 1) / args.steps) ** (1.0 if b2 else 0.9))
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_mult"]
        scaler.step(optimizer)
        scaler.update()
        if step % args.log_period == 0 or step == args.steps:
            LOGGER.info("source %s step %d/%d CE=%.5f lr=%.7f", args.task, step, args.steps, total, lr)
        if step % args.save_period == 0 or step == args.steps:
            destination = (args.output_dir / 'last.pth' if b2 else
                           (args.output_dir / 'source_last.pth' if step == args.steps else
                            args.output_dir / f'source_iter{step:06d}.pth'))
            write_checkpoint(destination, payload(step))
        if step == args.steps or (args.eval_period and step % args.eval_period == 0):
            observe(step)


def adapt(args, spec, device):
    from core.configs import cfg
    from train_ReMask import _apply_loss_weight_floor, _compute_reliability

    classes, _, target_split, eval_split = spec
    protocol = json.loads((args.output_dir / "protocol_adapt.json").read_text(encoding="utf-8"))
    target = ManifestDataset(args.data_root, target_split, "target", classes,
                             args.crop_size, args.train_policy)
    def evaluate_target(iteration: int, final: bool = False):
        evaluation = ManifestDataset(args.data_root, eval_split, "eval", classes,
                                     args.crop_size, args.train_policy)
        result = evaluate(teacher, classifier, evaluation, args, device)
        if final:
            summary = {
                "stage": "adapt", "iteration": iteration, "num_classes": classes,
                "seed": args.seed, "train_policy": args.train_policy,
                "normalization": "imagenet_rgb_planes", "crop_size": args.crop_size,
                "training_settings": training_settings,
                "source_checkpoint_sha256": protocol["files_sha256"]["source_checkpoint"],
            }
            export_result(args, result, summary, args.output_dir / "srmp_last.pth", protocol)
        return result
    if len(target) < args.batch_size:
        raise ValueError("Target dataset is smaller than one batch")
    checkpoint = load_checkpoint(args.source_checkpoint)
    if checkpoint.get("task") != args.task or checkpoint.get("num_classes") != classes:
        raise ValueError("Source checkpoint task/class count does not match target task")
    encoder, classifier = make_model(classes)
    encoder.load_state_dict(checkpoint["feature_extractor"], strict=True)
    classifier.load_state_dict(checkpoint["classifier"], strict=True)
    classifier.requires_grad_(False).eval()
    teacher = copy.deepcopy(encoder).requires_grad_(False).eval()
    rebuilder = SpatialRebuilder(2048, args.hidden_dim, args.depth)
    _, gram = source_kernel_geometry(classifier)
    encoder, classifier, teacher, rebuilder = (module.to(device) for module in
                                                (encoder, classifier, teacher, rebuilder))
    gram = gram.to(device)
    settings = SimpleNamespace(OBJECTIVE="relation", WEIGHT=args.relation_weight,
                               WARMUP_ITERS=args.warmup_steps, RAMP_ITERS=args.ramp_steps)
    config = cfg.clone()
    config.defrost()
    config.MODEL.NUM_CLASSES = classes
    config.freeze()
    optimizer = torch.optim.SGD(list(encoder.parameters()) + list(rebuilder.parameters()),
                                lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    uncertainty = {key: None for key in ("entropy_mean", "entropy_var", "kl_mean", "kl_var")}
    training_settings = {
        "task": args.task, "seed": args.seed, "steps": args.steps, "batch_size": args.batch_size,
        "accum_steps": args.accum_steps, "crop_size": args.crop_size, "lr": args.lr,
        "train_policy": args.train_policy,
        "relation_weight": args.relation_weight, "mask_ratio": args.mask_ratio,
        "block_size": args.block_size, "hidden_dim": args.hidden_dim,
        "depth": args.depth, "temperature": args.temperature,
        "warmup_steps": args.warmup_steps, "ramp_steps": args.ramp_steps,
        "ema_decay": args.ema_decay, "amp": args.amp,
    }
    start_step = 0
    if args.resume_training:
        resume = load_checkpoint(args.resume_training)
        if resume.get("stage") != "adapt" or resume.get("training_settings") != training_settings:
            raise ValueError("Resume checkpoint task or training settings do not match this run")
        if (resume.get("source_checkpoint_sha256") !=
                protocol["files_sha256"]["source_checkpoint"] or
                resume.get("target_manifest_sha256") !=
                protocol["files_sha256"]["manifest"]):
            raise ValueError("Resume checkpoint source model or target manifest changed")
        if not torch.allclose(resume["source_gram"].cpu(), gram.cpu(), atol=1e-6):
            raise ValueError("Resume checkpoint uses a different source classifier")
        encoder.load_state_dict(resume["student_feature_extractor"], strict=True)
        teacher.load_state_dict(resume["feature_extractor"], strict=True)
        rebuilder.load_state_dict(resume["srmp_rebuilder"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        uncertainty = {key: value.to(device) if value is not None else None
                       for key, value in resume["uncertainty"].items()}
        start_step = int(resume["iteration"])
        restore_rng(resume["rng"])
        LOGGER.info("Resuming %s at update %d/%d", args.task, start_step, args.steps)
    batches = infinite_batches(loader(target, args, shuffle=True))
    encoder.train()
    rebuilder.train()
    if start_step >= args.steps:
        LOGGER.info("Adaptation already reached its endpoint; evaluating saved teacher")
        LOGGER.info("final evaluation %s", json.dumps(evaluate_target(start_step, final=True)))
        return
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        totals = {"ce": 0.0, "relation": 0.0}
        weight = auxiliary_weight(step - 1, settings)
        for _ in range(args.accum_steps):
            clean, augmented, valid = next(batches)
            clean, augmented = clean.to(device), augmented.to(device)
            valid = valid.to(device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                teacher_features = teacher(clean)[1]
                teacher_prob = classifier(teacher_features, clean.shape[-2:]).float().softmax(1)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                features = encoder(augmented)[1]
                logits = classifier(features, augmented.shape[-2:])
            with torch.no_grad():
                pseudo = teacher_prob.argmax(1)
                reliability = _compute_reliability(
                    config, teacher_prob, logits.detach().float().softmax(1), uncertainty
                )
                pixel_weight = _apply_loss_weight_floor(config, reliability).squeeze(1)
                pixel_weight = pixel_weight * valid.float()
            ce = F.cross_entropy(logits.float(), pseudo, reduction="none")
            loss_ce = (ce * pixel_weight).sum() / pixel_weight.sum().clamp_min(1e-6)
            loss_rel = loss_ce.new_zeros(())
            if weight > 0:
                with torch.no_grad():
                    q = (classifier(teacher_features).float() / args.temperature).softmax(1)
                hidden = sample_block_mask(features.shape[0], *features.shape[-2:],
                                           args.mask_ratio, args.block_size, device)
                rebuilt = rebuilder(features, hidden)
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    predicted = classifier(rebuilt)
                p = (predicted.float() / args.temperature).softmax(1)
                valid_feature = F.adaptive_avg_pool2d(
                    valid.unsqueeze(1).float(), features.shape[-2:]
                ) >= 1.0 - 1e-6
                loss_rel = masked_mean(relational_cosine_map(p, q, gram),
                                       hidden[:, 0] & valid_feature[:, 0])
            loss = loss_ce + weight * loss_rel
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at adaptation step {step}")
            scaler.scale(loss / args.accum_steps).backward()
            totals["ce"] += float(loss_ce.detach()) / args.accum_steps
            totals["relation"] += float(loss_rel.detach()) / args.accum_steps
        lr = args.lr * (1 - (step - 1) / args.steps) ** 0.9
        for group in optimizer.param_groups:
            group["lr"] = lr
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            update_ema(teacher, encoder, args.ema_decay)
        if step % args.log_period == 0 or step == args.steps:
            LOGGER.info("adapt %s step %d/%d CE=%.5f relation=%.5f weight=%.4f lr=%.7f",
                        args.task, step, args.steps, totals["ce"], totals["relation"], weight, lr)
        if step % args.save_period == 0 or step == args.steps:
            write_checkpoint(args.output_dir / "srmp_last.pth", {
                "task": args.task, "num_classes": classes, "stage": "adapt", "iteration": step,
                "feature_extractor": teacher.state_dict(), "classifier": classifier.state_dict(),
                "student_feature_extractor": encoder.state_dict(),
                "srmp_rebuilder": rebuilder.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "source_gram": gram.cpu(),
                "uncertainty": uncertainty, "rng": rng_state(),
                "training_settings": training_settings,
                "crop_size": args.crop_size, "normalization": "imagenet_rgb_planes",
                "train_policy": args.train_policy,
                "seed": args.seed, "target_split": target_split,
                "target_manifest_sha256": protocol["files_sha256"]["manifest"],
                "source_checkpoint_sha256": protocol["files_sha256"]["source_checkpoint"],
            })
        if args.eval_period and step % args.eval_period == 0:
            result = evaluate_target(step, final=step == args.steps)
            LOGGER.info("eval step %d %s", step, json.dumps(result))
            encoder.train()
            rebuilder.train()
    if not args.eval_period or args.steps % args.eval_period:
        LOGGER.info("final evaluation %s", json.dumps(evaluate_target(args.steps, final=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', choices=('legacy', B2_PROTOCOL), default='legacy')
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--stage", choices=("source", "adapt", "eval"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/remote_sensing/dgle"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--source-baseline", type=Path,
                        help="Source-only metrics_eval.json to include a verified comparison in the portable result")
    parser.add_argument("--backbone-weights", type=Path)
    parser.add_argument("--resume-training", type=Path,
                        help="Continue adaptation from srmp_last.pth with matching settings")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--train-policy", choices=("rs_standard", "basic", "siamseg_isprs"),
                        default="rs_standard")
    parser.add_argument("--head-lr-mult", type=float, default=1., help="Source ASPP LR multiplier")
    parser.add_argument("--source-val-split", choices=("isprs_vh_irrg_val",),
                        help="Select source_best using held-out source tiles; never target labels")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-period", type=int, default=20)
    parser.add_argument("--save-period", type=int, default=1000)
    parser.add_argument("--eval-period", type=int, default=0,
                        help="0 evaluates only at the endpoint; no checkpoint is selected by target labels")
    parser.add_argument('--eval-limit', type=int, default=0, help='Smoke-only maximum evaluation patches')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--no-eval', action='store_true', help='B2 smoke diagnostic without validation')
    parser.add_argument("--relation-weight", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--ramp-steps", type=int, default=1000)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    args = parser.parse_args()
    if args.protocol == B2_PROTOCOL:
        if args.task != 'vh2pd' or args.stage == 'adapt':
            parser.error('B2 protocol supports vh2pd source/eval; use train_SND_remote.py for adaptation')
        args.data_root = Path('data/remote_sensing/siamseg_sfuda_b2_v1') if args.data_root == Path('data/remote_sensing/dgle') else args.data_root
        if args.stage == 'source':
            args.source_val_split = None if args.no_eval else 'source_val'
            args.train_policy = 'siamseg_sfuda'
            if args.eval_period == 0 and not args.no_eval:
                args.eval_period = 500
            if args.head_lr_mult == 1:
                args.head_lr_mult = 10.
            if args.lr == 2.5e-4:
                args.lr = 6e-5
        if args.eval_limit and not args.smoke_only:
            parser.error('--eval-limit is reserved for smoke runs')
        if args.no_eval and not args.smoke_only:
            parser.error('--no-eval is reserved for B2 smoke runs')
        if args.smoke_only and args.stage == 'source' and args.steps > 10:
            parser.error('B2 smoke runs are limited to 10 optimizer updates')
        if args.stage == 'source':
            expected_sha = json.loads((Path(__file__).parent/'configs/remote_sensing/siamseg_sfuda_b2_v1.json').read_text())['pretrain_sha256']
            if args.backbone_weights and args.backbone_weights.is_file() and sha256_file(args.backbone_weights) != expected_sha:
                parser.error('MiT-B2 ImageNet checkpoint SHA-256 mismatch')
    if min(args.steps, args.batch_size, args.accum_steps, args.crop_size,
           args.log_period, args.save_period, args.block_size, args.hidden_dim, args.depth) < 1:
        parser.error("Step, batch, crop, logging, and rebuilder dimensions must be positive")
    if args.workers < 0 or args.eval_period < 0 or args.lr <= 0 or args.temperature <= 0 or args.head_lr_mult <= 0:
        parser.error("Invalid worker, evaluation, learning-rate, or temperature setting")
    if not 0 < args.mask_ratio < 1 or not 0 <= args.ema_decay < 1:
        parser.error("Require 0 < mask-ratio < 1 and 0 <= ema-decay < 1")
    if args.stage == "source" and (not args.backbone_weights or not args.backbone_weights.is_file()):
        parser.error("Source training requires a local --backbone-weights ResNet101 checkpoint")
    if args.stage in ("adapt", "eval") and (not args.source_checkpoint or not args.source_checkpoint.is_file()):
        parser.error("Adapt/eval requires --source-checkpoint from source training")
    if args.resume_training and (not args.resume_training.is_file() or
                                 (args.stage != 'adapt' and not (args.stage == 'source' and args.protocol == B2_PROTOCOL))):
        parser.error("--resume-training requires a supported stage and an existing checkpoint")
    if args.source_val_split and (args.stage != "source" or args.task != "vh2pd" or not args.eval_period):
        parser.error("Source validation requires vh2pd source stage and positive --eval-period")
    if args.train_policy == "siamseg_isprs" and (args.stage != "source" or args.task != "vh2pd"):
        parser.error("siamseg_isprs is the labeled VH source pretraining recipe")
    if args.stage == "source" and not args.resume_training and (any(args.output_dir.glob("*.pth")) or
                                   (args.output_dir / "protocol_source.json").exists()):
        parser.error("Source run already exists; use a new output directory to preserve results")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = B2_SPEC if args.protocol == B2_PROTOCOL else TASKS[args.task]
    record_protocol(args, spec)
    if args.stage == "source":
        source_train(args, spec, device)
    elif args.stage == "adapt":
        adapt(args, spec, device)
    else:
        classes, _, _, eval_split = spec
        checkpoint = load_checkpoint(args.source_checkpoint)
        if checkpoint.get("task") != args.task or checkpoint.get("num_classes") != classes:
            raise ValueError("Checkpoint task/class count mismatch")
        if args.protocol == B2_PROTOCOL and (checkpoint.get('architecture') != 'segformer_mit_b2_nvidia_mlp768' or
               checkpoint.get('protocol_sha256') != sha256_file(Path(__file__).parent / 'configs/remote_sensing/siamseg_sfuda_b2_v1.json')):
            raise ValueError('B2 checkpoint architecture/protocol mismatch')
        encoder, classifier = make_model(classes, architecture=checkpoint.get('architecture', 'resnet101_aspp'))
        encoder.load_state_dict(checkpoint["feature_extractor"], strict=True)
        classifier.load_state_dict(checkpoint["classifier"], strict=True)
        dataset = ManifestDataset(args.data_root, eval_split, "eval", classes, args.crop_size)
        metrics = evaluate(encoder.to(device), classifier.to(device), dataset, args, device)
        result = {
            "task": args.task,
            "split": eval_split,
            "source_checkpoint_sha256": sha256_file(args.source_checkpoint),
            "metrics": metrics,
        }
        if args.protocol == B2_PROTOCOL:
            result.update({key: checkpoint[key] for key in
                           ('architecture', 'protocol', 'protocol_sha256', 'target_eval_manifest_sha256')})
        (args.output_dir / "metrics_eval.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        export_result(args, metrics, checkpoint, args.source_checkpoint)


if __name__ == "__main__":
    main()
