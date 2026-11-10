"""Read-only SRMP diagnostics on a fixed, complete target-training cohort.

Direction 1: measure error detection, not assume agreement implies correctness.
Direction 2 audit: independently frozen DINO features define region proposals
and cross-image neighbours; teacher predictions NEVER construct that reference.
DINO regions are feature-coherent proposals, not guaranteed object instances or
semantic class labels. Ground truth is used only to score the diagnostics.
"""

import csv
import gc
import hashlib
import json
import logging
import math
import os
import random
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def add_diagnostic_config(config):
    from yacs.config import CfgNode as CN

    config.SRMP_DIAG = CN()
    d = config.SRMP_DIAG
    d.ENABLED = False  # Enabled by the direction-2 SLURM launcher.
    d.PERIOD = 3000
    d.INITIAL = True
    d.MAX_IMAGES = 0  # 0 = every unique image, not the rare-class training sampler.
    d.INPUT_SIZE = (1024, 512)  # width, height; fixed clean view.
    d.MASK_REPEATS = 2
    d.SEED = 2026
    d.HIGH_CONFIDENCE = 0.9
    d.HISTOGRAM_BINS = 256
    d.REQUIRE_GT = True
    d.DINO = True
    d.DINO_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
    d.DINO_REVISION = "main"
    d.LOCAL_FILES_ONLY = False
    d.CACHE_DIR = ""
    d.DINO_LONG_SIDE = 768
    d.REGION_CLUSTERS = 12
    d.KMEANS_ITERS = 15
    d.MIN_REGION_PATCHES = 2
    d.MAX_REGIONS_PER_IMAGE = 64
    d.NEIGHBOUR_BANK_SIZE = 16384
    d.NEIGHBOURS = 3
    d.MIN_SIMILARITY = 0.85
    d.GT_REGION_PURITY = 0.8


def validate_diagnostic_config(config):
    d = config.SRMP_DIAG
    if not d.ENABLED:
        return
    if min(d.PERIOD, *d.INPUT_SIZE, d.MASK_REPEATS, d.HISTOGRAM_BINS,
           d.DINO_LONG_SIDE, d.REGION_CLUSTERS, d.KMEANS_ITERS,
           d.MIN_REGION_PATCHES, d.MAX_REGIONS_PER_IMAGE,
           d.NEIGHBOUR_BANK_SIZE, d.NEIGHBOURS) < 1 or d.MAX_IMAGES < 0:
        raise ValueError("SRMP_DIAG sizes/counts must be positive; MAX_IMAGES may be zero (all).")
    if not 0 <= d.HIGH_CONFIDENCE <= 1 or not 0 <= d.GT_REGION_PURITY <= 1:
        raise ValueError("Diagnostic confidence and purity thresholds must be in [0, 1].")
    if not -1 <= d.MIN_SIMILARITY <= 1:
        raise ValueError("SRMP_DIAG.MIN_SIMILARITY must be in [-1, 1].")
    if d.DINO and not d.DINO_MODEL:
        raise ValueError("Specify a pretrained DINO model or local Hugging Face model directory.")


def divide(a, b):
    return float(a / b) if b else None


def save_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def load_state(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_state(value, path):
    temporary = str(path) + ".tmp"
    torch.save(value, temporary)
    os.replace(temporary, path)


@contextmanager
def preserve_training_state(*modules):
    """Audits must not change training RNG streams, BN state, dropout or modes."""
    modes = [(child, child.training) for module in modules for child in module.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        for module in modules:
            module.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def js_map(p, q):
    p, q = p.float(), q.float()
    middle = (p + q) * 0.5
    return (0.5 * (p * (p.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log()) +
                   q * (q.clamp_min(1e-8).log() - middle.clamp_min(1e-8).log())).sum(1) / math.log(2)).clamp(0, 1)


def histogram_metrics(counts, errors):
    """High score predicts an error. Binned AUROC/AP and low-score risk curves."""
    counts, errors = np.asarray(counts, dtype=float), np.asarray(errors, dtype=float)
    negatives = counts - errors
    positive, negative = errors.sum(), negatives.sum()
    auc = divide((errors * (np.cumsum(negatives) - negatives + 0.5 * negatives)).sum(), positive * negative)
    tp, total = np.cumsum(errors[::-1]), np.cumsum(counts[::-1])
    precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
    ap = divide((precision * errors[::-1]).sum(), positive)
    risks = {}
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        budget = counts.sum() * fraction
        taken = np.clip(budget - (np.cumsum(counts) - counts), 0, counts)
        error_rate = np.divide(errors, counts, out=np.zeros_like(errors), where=counts > 0)
        risks[str(fraction)] = divide((taken * error_rate).sum(), taken.sum())
    return {"count": int(counts.sum()), "error_rate": divide(positive, counts.sum()),
            "error_detection_auroc_binned": auc, "error_detection_ap_binned": ap,
            "risk_at_retained_fraction": risks}


class ErrorProbe:
    def __init__(self, classes=19, bins=256, high_confidence=0.9):
        self.classes, self.bins, self.high_confidence = classes, bins, high_confidence
        self.tables = {name: np.zeros((classes, 2, 2), dtype=np.int64)
                       for name in ("teacher_masked", "student_masked")}
        self.histograms = {}
        self.confidence_table = np.zeros((10, 2, 2), dtype=np.int64)
        self.image_rows = []

    def update(self, name, gt, teacher, masked, student, selected, relation, repeat_disagreement=None):
        confidence, label = teacher.max(1)
        wrong = label != gt
        valid = selected & (gt >= 0) & (gt < self.classes)
        agreements = {"teacher_masked": masked.argmax(1) == label,
                      "student_masked": masked.argmax(1) == student.argmax(1)}
        for key, agreement in agreements.items():
            # axes: pseudo class, agreement (0/1), teacher pseudo-label wrong (0/1).
            ids = label[valid] * 4 + agreement[valid].long() * 2 + wrong[valid].long()
            self.tables[key] += torch.bincount(ids, minlength=self.classes * 4).view(self.classes, 2, 2).cpu().numpy()
        confidence_bins = (confidence[valid] * 10).long().clamp(0, 9)
        ids = confidence_bins * 4 + agreements["teacher_masked"][valid].long() * 2 + wrong[valid].long()
        self.confidence_table += torch.bincount(ids, minlength=40).view(10, 2, 2).cpu().numpy()
        scores = {
            "teacher_masked_js": js_map(teacher, masked),
            "student_masked_js": js_map(student, masked),
            "teacher_student_js": js_map(teacher, student),
            "source_relation": (relation / 2).clamp(0, 1),
            "one_minus_confidence": 1 - confidence,
            "teacher_entropy": -(teacher * teacher.clamp_min(1e-8).log()).sum(1) / math.log(self.classes),
        }
        if repeat_disagreement is not None:
            scores["masked_repeat_disagreement"] = repeat_disagreement
        for stratum, selection in (("all", valid), ("high_confidence", valid & (confidence >= self.high_confidence))):
            for key, score in scores.items():
                bins = (score[selection].clamp(0, 1) * (self.bins - 1)).long()
                n = torch.bincount(bins, minlength=self.bins).cpu().numpy()
                e = torch.bincount(bins[wrong[selection]], minlength=self.bins).cpu().numpy()
                hist = self.histograms.setdefault(stratum + "/" + key, np.zeros((2, self.bins), dtype=np.int64))
                hist[0] += n
                hist[1] += e
        agree = agreements["teacher_masked"]
        row = {"image": name, "selected_gt_cells": int(valid.sum()), "wrong": int((wrong & valid).sum()),
               "agree_wrong": int((agree & wrong & valid).sum()), "agree_correct": int((agree & ~wrong & valid).sum()),
               "disagree_wrong": int((~agree & wrong & valid).sum()), "disagree_correct": int((~agree & ~wrong & valid).sum()),
               "high_conf_wrong": int((valid & wrong & (confidence >= self.high_confidence)).sum()),
               "high_conf_agree_wrong": int((valid & wrong & agree & (confidence >= self.high_confidence)).sum())}
        self.image_rows.append(row)

    def summary(self):
        result = {"table_axes": ["agreement: 0=no, 1=yes", "pseudo_label_error: 0=correct, 1=wrong"],
                  "score_orientation": "higher = more likely pseudo-label error",
                  "histogram_bins": self.bins, "high_confidence_threshold": self.high_confidence,
                  "images": len(self.image_rows), "agreement": {}, "score_metrics": {}, "histograms": {},
                  "confidence_stratified_agreement": []}
        for index, table in enumerate(self.confidence_table):
            result["confidence_stratified_agreement"].append({
                "confidence_interval": [index / 10, (index + 1) / 10], "table": table.tolist(),
                "p_error_given_agreement": divide(table[1, 1], table[1].sum()),
                "p_error_given_disagreement": divide(table[0, 1], table[0].sum())})
        for key, table in self.tables.items():
            t = table.sum(0)
            result["agreement"][key] = {
                "table": t.tolist(), "per_pseudo_class_table": table.tolist(),
                "p_error_given_agreement": divide(t[1, 1], t[1].sum()),
                "p_error_given_disagreement": divide(t[0, 1], t[0].sum()),
                "error_recall_of_disagreement": divide(t[0, 1], t[:, 1].sum()),
                "false_alarm_rate_of_disagreement": divide(t[0, 0], t[:, 0].sum()),
                "consistent_wrong_fraction_of_errors": divide(t[1, 1], t[:, 1].sum()),
            }
        for key, (n, e) in self.histograms.items():
            result["score_metrics"][key] = histogram_metrics(n, e)
            result["histograms"][key] = {"count": n.tolist(), "errors": e.tolist()}
        return result


def discover_regions(features, clusters=12, iterations=15, min_size=2, max_regions=64, seed=0):
    """Spherical k-means + 4-connected components using ONLY frozen DINO tokens."""
    height, width, dim = features.shape
    flat = F.normalize(features.reshape(-1, dim).float(), dim=1)
    count = min(clusters, len(flat))
    generator = torch.Generator(device=flat.device).manual_seed(seed)
    first = int(torch.randint(len(flat), (1,), generator=generator, device=flat.device))
    centers = [flat[first]]
    nearest = flat @ centers[0]
    for _ in range(count - 1):
        centers.append(flat[nearest.argmin()])
        nearest = torch.maximum(nearest, flat @ centers[-1])
    centers = torch.stack(centers)
    for _ in range(iterations):
        labels = (flat @ centers.T).argmax(1)
        sums = torch.zeros_like(centers).index_add_(0, labels, flat)
        nonempty = torch.bincount(labels, minlength=count) > 0
        centers = torch.where(nonempty[:, None], F.normalize(sums, dim=1), centers)
    labels = (flat @ centers.T).argmax(1).view(height, width).cpu().numpy()
    visited, components = np.zeros((height, width), bool), []
    for y in range(height):
        for x in range(width):
            if visited[y, x]:
                continue
            queue, pixels = deque([(y, x)]), []
            visited[y, x] = True
            while queue:
                yy, xx = queue.popleft()
                pixels.append(yy * width + xx)
                for ny, nx in ((yy - 1, xx), (yy + 1, xx), (yy, xx - 1), (yy, xx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and labels[ny, nx] == labels[y, x]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(pixels) >= min_size:
                components.append(pixels)
    components = sorted(components, key=len, reverse=True)[:max_regions]
    region_map = torch.full((height * width,), -1, dtype=torch.int32)
    embeddings = []
    for region_id, pixels in enumerate(components):
        region_map[pixels] = region_id
        embeddings.append(F.normalize(flat[pixels].mean(0), dim=0).cpu())
    return region_map.view(height, width), (torch.stack(embeddings) if embeddings else torch.empty(0, dim))


def neighbour_graph(embeddings, image_ids, bank_size, k, seed, device):
    """Every region queries a bounded reference bank; same-image matches excluded."""
    count = len(embeddings)
    if count == 0:
        return {"index": torch.empty(0, k, dtype=torch.long), "similarity": torch.empty(0, k),
                "random_index": torch.empty(0, dtype=torch.long), "bank_size": 0}
    generator = torch.Generator().manual_seed(seed)
    bank = torch.randperm(count, generator=generator)[:bank_size]
    bank_features = F.normalize(embeddings[bank].float(), dim=1).to(device)
    bank_images = image_ids[bank].to(device)
    neighbours = torch.full((count, k), -1, dtype=torch.long)
    similarities = torch.full((count, k), -1.0)
    for start in range(0, count, 128):
        end = min(start + 128, count)
        score = F.normalize(embeddings[start:end].float(), dim=1).to(device) @ bank_features.T
        score.masked_fill_(image_ids[start:end, None].to(device) == bank_images[None], -float("inf"))
        values, ids = score.topk(min(k, len(bank)), dim=1)
        good = torch.isfinite(values.cpu())
        neighbours[start:end, :values.shape[1]] = torch.where(good, bank[ids.cpu()], -1)
        similarities[start:end, :values.shape[1]] = torch.where(good, values.cpu(), -1.0)
        if end % 2048 == 0 or end == count:
            logging.getLogger("DTST.diagnostics").info("DINO neighbour graph: %d/%d regions", end, count)
    # Uniform region reference conditional on a different image, for a null baseline.
    random_ids = torch.full((count,), -1, dtype=torch.long)
    for image in torch.unique(image_ids):
        query = torch.where(image_ids == image)[0]
        foreign = torch.where(image_ids != image)[0]
        if len(foreign):
            random_ids[query] = foreign[torch.randint(len(foreign), (len(query),), generator=generator)]
    return {"index": neighbours, "similarity": similarities, "random_index": random_ids, "bank_size": len(bank)}


class DinoReference:
    """Pretrained backbone only: no supervised segmentation/classification head."""
    def __init__(self, settings, device):
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("Install requirements_srmp_diagnostics.txt to enable the frozen DINO audit.") from exc
        options = {"local_files_only": settings.LOCAL_FILES_ONLY, "trust_remote_code": False}
        if not Path(settings.DINO_MODEL).is_dir():
            options["revision"] = settings.DINO_REVISION
        try:
            self.processor = AutoImageProcessor.from_pretrained(settings.DINO_MODEL, **options)
            self.model = AutoModel.from_pretrained(settings.DINO_MODEL, **options).to(device).eval().requires_grad_(False)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "Cannot load pretrained DINO. DINOv3 requires approved weight access; use an authorized "
                "HF account/token or set SRMP_DIAG.DINO_MODEL to a local Hugging Face model directory. "
                "No random-weight or alternative-model fallback is used."
            ) from exc
        if self.model.config.model_type not in ("dinov3_vit", "dinov2", "dinov2_with_registers"):
            raise ValueError("Choose a DINOv3 ViT or DINOv2 backbone, not a supervised task head.")
        self.device, self.settings = device, settings
        self.patch = int(self.model.config.patch_size)
        self.metadata = {"model": settings.DINO_MODEL, "revision": settings.DINO_REVISION,
                         "resolved_commit": getattr(self.model.config, "_commit_hash", None),
                         "model_type": self.model.config.model_type, "patch_size": self.patch,
                         "supervision": "pretrained self-supervised backbone; no task head"}

    @torch.no_grad()
    def features(self, image):
        scale = self.settings.DINO_LONG_SIDE / max(image.size)
        width, height = [max(self.patch, round(v * scale / self.patch) * self.patch) for v in image.size]
        rgb = np.asarray(image.resize((width, height), Image.BICUBIC), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mean = tensor.new_tensor(self.processor.image_mean).view(1, 3, 1, 1)
        std = tensor.new_tensor(self.processor.image_std).view(1, 3, 1, 1)
        # Preserve aspect ratio and the entire image. No default square center crop.
        output = self.model(pixel_values=(tensor - mean) / std).last_hidden_state
        tokens = output[:, 1 + getattr(self.model.config, "num_register_tokens", 0):]
        if tokens.shape[1] != (height // self.patch) * (width // self.patch):
            raise ValueError("Unexpected DINO patch-token count; CLS/register tokens must be excluded.")
        if not torch.isfinite(tokens).all():
            raise FloatingPointError("DINO returned non-finite features; cannot build a trustworthy reference bank.")
        return F.normalize(tokens[0].float(), dim=-1).view(height // self.patch, width // self.patch, -1)


def target_records(config):
    from core.datasets.dataset_path_catalog import DatasetCatalog

    specification = DatasetCatalog.DATASETS[config.DATASETS.TARGET_TRAIN]
    root = Path(DatasetCatalog.DATASET_DIR) / specification["data_dir"]
    listing = Path(DatasetCatalog.DATASET_DIR) / specification["data_list"]
    image_folder = "leftImg8bit" if (root / "leftImg8bit").is_dir() else "data"
    label_folder = "gtFine" if (root / "gtFine").is_dir() else "label"
    names = sorted(set(line.strip() for line in listing.read_text().splitlines() if line.strip()))
    if config.SRMP_DIAG.MAX_IMAGES:
        generator = random.Random(config.SRMP_DIAG.SEED)
        names = sorted(generator.sample(names, min(len(names), config.SRMP_DIAG.MAX_IMAGES)))
    records = []
    for name in names:
        if "/" not in name:
            name = name.split("_")[0] + "/" + name
        image = root / image_folder / "train" / name
        label = root / label_folder / "train" / (name.split("_leftImg8bit")[0] + "_gtFine_labelIds.png")
        if not image.is_file():
            raise FileNotFoundError("Diagnostic target image missing: {}".format(image))
        if config.SRMP_DIAG.REQUIRE_GT and not label.is_file():
            raise FileNotFoundError("Direction-1 error evaluation requires target GT (diagnostics only): {}".format(label))
        records.append({"name": name, "image": str(image), "label": str(label)})
    if not records:
        raise ValueError("The diagnostic target cohort is empty.")
    return records


def load_gt(record, size):
    """Cityscapes labelIds -> trainIds, nearest resize; never used by DINO/cache."""
    if not Path(record["label"]).is_file():
        return torch.full(size, 255, dtype=torch.long)
    with Image.open(record["label"]) as image:
        raw = np.asarray(image.resize((size[1], size[0]), Image.NEAREST))
    converted = np.full(raw.shape, 255, dtype=np.int64)
    for train_id, label_id in enumerate((7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)):
        converted[raw == label_id] = train_id
    return torch.from_numpy(converted)


def region_statistics(probability, region_map, gt, classes):
    grid = F.interpolate(probability.float(), region_map.shape, mode="bilinear", align_corners=False)[0].cpu()
    labels = grid.argmax(0)
    count = int(region_map.max()) + 1
    stats = {key: [] for key in ("probability", "label", "purity", "patches", "gt_label", "gt_purity", "gt_valid", "accuracy")}
    for region_id in range(count):
        pixels = region_map == region_id
        predicted = torch.bincount(labels[pixels], minlength=classes)
        valid = pixels & (gt >= 0) & (gt < classes)
        truth = torch.bincount(gt[valid], minlength=classes)
        stats["probability"].append(grid[:, pixels].mean(1))
        stats["label"].append(predicted.argmax())
        stats["purity"].append(predicted.max() / predicted.sum())
        stats["patches"].append(pixels.sum())
        stats["gt_label"].append(truth.argmax() if valid.any() else torch.tensor(-1))
        stats["gt_purity"].append(truth.max() / truth.sum().clamp_min(1))
        stats["gt_valid"].append(valid.sum())
        stats["accuracy"].append((labels[valid] == gt[valid]).sum() / valid.sum().clamp_min(1))
    return {key: torch.stack(values) if values else torch.empty((0, classes) if key == "probability" else (0,))
            for key, values in stats.items()}


def object_summary(stats, graph, minimum_similarity, gt_purity, previous=None):
    labels, probabilities = stats["label"].long(), stats["probability"].float()
    indices, similarity = graph["index"], graph["similarity"]
    valid = (indices >= 0) & (similarity >= minimum_similarity)
    queries = torch.arange(len(labels))[:, None].expand_as(indices)[valid]
    neighbours = indices[valid]
    agreeing = labels[queries] == labels[neighbours]
    marginal = torch.bincount(labels, minlength=probabilities.shape[1]).float() / max(1, len(labels))
    random_valid = graph["random_index"] >= 0
    random_agree = labels[random_valid] == labels[graph["random_index"][random_valid]]
    gt_ok = (stats["gt_valid"] > 0) & (stats["gt_purity"] >= gt_purity)
    pair_gt = gt_ok[queries] & gt_ok[neighbours]
    same_gt = stats["gt_label"][queries] == stats["gt_label"][neighbours]
    both_wrong = (labels[queries] != stats["gt_label"][queries]) & (labels[neighbours] != stats["gt_label"][neighbours])
    result = {
        "regions": len(labels), "neighbour_bank_regions": graph["bank_size"],
        "matched_cross_image_pairs": len(queries), "matched_query_regions": int(torch.unique(queries).numel()),
        "query_match_coverage": divide(torch.unique(queries).numel(), len(labels)),
        "within_region_majority_purity_macro": divide(stats["purity"].sum(), len(labels)),
        "within_region_majority_purity_patch_weighted": divide((stats["purity"] * stats["patches"]).sum(), stats["patches"].sum()),
        "teacher_cross_image_agreement": divide(agreeing.sum(), len(agreeing)),
        "random_cross_image_agreement": divide(random_agree.sum(), len(random_agree)),
        "teacher_class_marginal": marginal.tolist(),
        "teacher_class_entropy": float(-(marginal * marginal.clamp_min(1e-8).log()).sum() / math.log(len(marginal))),
        "teacher_distinct_region_classes": int((marginal > 0).sum()),
        "reference_pairs_with_pure_gt": int(pair_gt.sum()),
        "reference_pair_gt_same_class_rate": divide((same_gt & pair_gt).sum(), pair_gt.sum()),
        "teacher_pair_agreement_given_gt_same_class": divide((agreeing & same_gt & pair_gt).sum(), (same_gt & pair_gt).sum()),
        "teacher_pair_agreement_given_gt_different_class": divide((agreeing & ~same_gt & pair_gt).sum(), (~same_gt & pair_gt).sum()),
        "consistent_both_wrong_rate_on_gt_pairs": divide((agreeing & both_wrong & pair_gt).sum(), pair_gt.sum()),
        "region_proposal_gt_purity_macro": divide(stats["gt_purity"][stats["gt_valid"] > 0].sum(), (stats["gt_valid"] > 0).sum()),
        "teacher_region_gt_accuracy_patch_weighted": divide((stats["accuracy"] * stats["gt_valid"]).sum(), stats["gt_valid"].sum()),
        "temporal_label_flip_rate": None, "temporal_probability_js": None,
    }
    if previous is not None and len(labels):
        result["temporal_label_flip_rate"] = float((labels != previous["label"].long()).float().mean())
        result["temporal_probability_js"] = float(js_map(probabilities[:, :, None, None], previous["probability"].float()[:, :, None, None]).mean())
    return result


class TrainingDiagnostics:
    def __init__(self, config, device, records=None, reference_factory=DinoReference):
        self.config, self.settings, self.device = config, config.SRMP_DIAG, device
        self.records = target_records(config) if records is None else records
        self.reference_factory = reference_factory
        self.output = Path(config.OUTPUT_DIR) / "srmp_diagnostics"
        self.output.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("DTST.diagnostics")
        self.reference_files, self.reference_meta, self.graph = [], None, None
        self.region_image_ids = self.region_local_ids = None
        self.fingerprint = None

    def _prepare_reference(self):
        if not self.settings.DINO or self.graph is not None:
            return
        metadata = {
            "version": 1, "settings": {key: self.settings[key] for key in (
                "DINO_MODEL", "DINO_REVISION", "DINO_LONG_SIDE", "REGION_CLUSTERS", "KMEANS_ITERS",
                "MIN_REGION_PATCHES", "MAX_REGIONS_PER_IMAGE", "SEED", "NEIGHBOURS", "NEIGHBOUR_BANK_SIZE")},
            "images": [(r["name"], str(Path(r["image"]).resolve()), Path(r["image"]).stat().st_size,
                        Path(r["image"]).stat().st_mtime_ns) for r in self.records],
        }
        local_model = Path(self.settings.DINO_MODEL)
        if local_model.is_dir():
            metadata["local_model_files"] = [(str(p.relative_to(local_model)), p.stat().st_size, p.stat().st_mtime_ns)
                                              for p in sorted(local_model.rglob("*")) if p.is_file()]
        self.fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:24]
        cache = Path(self.settings.CACHE_DIR or str(self.output / "reference")) / self.fingerprint
        cache.mkdir(parents=True, exist_ok=True)
        self.reference_files = [cache / (hashlib.sha256(r["name"].encode()).hexdigest()[:24] + ".pth") for r in self.records]
        reference = None
        for index, (record, path) in enumerate(zip(self.records, self.reference_files)):
            if path.exists():
                continue
            if reference is None:
                self.logger.info("Preparing frozen DINO reference for %d target images (cached once).", len(self.records))
                reference = self.reference_factory(self.settings, self.device)
            with Image.open(record["image"]) as raw:
                features = reference.features(raw.convert("RGB"))
            region_map, embeddings = discover_regions(
                features, self.settings.REGION_CLUSTERS, self.settings.KMEANS_ITERS,
                self.settings.MIN_REGION_PATCHES, self.settings.MAX_REGIONS_PER_IMAGE, self.settings.SEED + index)
            save_state({"region_map": region_map, "embeddings": embeddings.half(), "model": reference.metadata}, path)
            del features
            if (index + 1) % 25 == 0 or index + 1 == len(self.records):
                self.logger.info("DINO reference: %d/%d images", index + 1, len(self.records))
        if reference is not None:
            del reference
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        descriptors, image_ids, local_ids = [], [], []
        for index, path in enumerate(self.reference_files):
            item = load_state(path)
            descriptors.append(item["embeddings"])
            image_ids.extend([index] * len(item["embeddings"]))
            local_ids.extend(range(len(item["embeddings"])))
            if self.reference_meta is not None and self.reference_meta != item["model"]:
                raise ValueError("The reference cache mixes different DINO model revisions; rebuild it with a pinned revision.")
            self.reference_meta = item["model"]
        self.region_image_ids = torch.tensor(image_ids, dtype=torch.long)
        self.region_local_ids = torch.tensor(local_ids, dtype=torch.long)
        graph_path = cache / "neighbour_graph.pth"
        if graph_path.exists():
            self.graph = load_state(graph_path)
        else:
            self.graph = neighbour_graph(torch.cat(descriptors), self.region_image_ids,
                                         self.settings.NEIGHBOUR_BANK_SIZE, self.settings.NEIGHBOURS,
                                         self.settings.SEED, self.device)
            save_state(self.graph, graph_path)
        save_json({**metadata, "fingerprint": self.fingerprint, "backbone": self.reference_meta}, cache / "manifest.json")

    def _previous(self, iteration):
        for path in sorted(self.output.glob("regions_iter*.pth"), reverse=True):
            step = int(path.stem.split("iter")[-1])
            if step < iteration and (self.output / "audit_iter{:06d}.json".format(step)).exists():
                state = load_state(path)
                if (state["fingerprint"] == self.fingerprint
                        and state.get("input_size") == list(self.settings.INPUT_SIZE)):
                    return state, step
        return None, None

    @torch.no_grad()
    def run(self, iteration, encoder, classifier, teacher, rebuilder, gram, force=False):
        from train_SRMP import relational_cosine_map, sample_block_mask

        if not force and iteration % self.settings.PERIOD:
            return
        started = time.time()
        with preserve_training_state(encoder, classifier, teacher, rebuilder):
            self._prepare_reference()
            probe = ErrorProbe(self.config.MODEL.NUM_CLASSES, self.settings.HISTOGRAM_BINS, self.settings.HIGH_CONFIDENCE)
            all_stats, region_rows = {}, []
            retained_patches = total_patches = 0
            for index, record in enumerate(self.records):
                with Image.open(record["image"]) as raw:
                    rgb = np.asarray(raw.convert("RGB").resize(tuple(self.settings.INPUT_SIZE), Image.BICUBIC), dtype=np.float32) / 255
                image = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(self.device)
                if self.config.INPUT.TO_BGR255:
                    image = image[:, [2, 1, 0]] * 255
                image = (image - image.new_tensor(self.config.INPUT.PIXEL_MEAN)[None, :, None, None]) / image.new_tensor(self.config.INPUT.PIXEL_STD)[None, :, None, None]
                q = classifier(teacher(image)[1]).float().softmax(1)
                features = encoder(image)[1]
                clean = classifier(features).float().softmax(1)
                accumulator, observations = torch.zeros_like(q), torch.zeros_like(q[:, :1])
                teacher_survival = torch.zeros_like(observations)
                # The same image gets the same masks at every snapshot. Entire
                # audit RNG state is restored before returning to training.
                seed = self.settings.SEED + int(hashlib.sha256(record["name"].encode()).hexdigest()[:8], 16)
                torch.manual_seed(seed)
                for _ in range(self.settings.MASK_REPEATS):
                    hidden = sample_block_mask(1, *features.shape[-2:], self.config.SRMP.MASK_RATIO,
                                               self.config.SRMP.BLOCK_SIZE, self.device)
                    p = classifier(rebuilder(features, hidden)).float().softmax(1)
                    accumulator += p * hidden
                    observations += hidden
                    teacher_survival += (p.argmax(1) == q.argmax(1))[:, None] * hidden
                masked = accumulator / observations.clamp_min(1)
                selected = observations[:, 0] > 0
                truth = load_gt(record, q.shape[-2:]).to(self.device)[None]
                probe.update(record["name"], truth, q, masked, clean, selected,
                             relational_cosine_map(masked, q, gram),
                             1 - (teacher_survival / observations.clamp_min(1))[:, 0])
                if self.settings.DINO:
                    reference = load_state(self.reference_files[index])
                    region_map = reference["region_map"].long()
                    statistics = region_statistics(q, region_map, load_gt(record, region_map.shape), self.config.MODEL.NUM_CLASSES)
                    for key, value in statistics.items():
                        all_stats.setdefault(key, []).append(value)
                    retained_patches += int((region_map >= 0).sum())
                    total_patches += region_map.numel()
                    for region_id in range(len(statistics["label"])):
                        row = {"image": record["name"], "region_id": region_id,
                               "teacher_label": int(statistics["label"][region_id]),
                               "teacher_majority_fraction": float(statistics["purity"][region_id]),
                               "patches": int(statistics["patches"][region_id]),
                               "gt_majority_label": int(statistics["gt_label"][region_id]),
                               "gt_purity": float(statistics["gt_purity"][region_id]),
                               "gt_valid_patches": int(statistics["gt_valid"][region_id]),
                               "teacher_gt_accuracy": float(statistics["accuracy"][region_id]) if statistics["gt_valid"][region_id] else None}
                        region_rows.append(row)
                del features, q, clean, accumulator, observations, p, image, masked
                if (index + 1) % 25 == 0 or index + 1 == len(self.records):
                    self.logger.info("SRMP audit iter %d: %d/%d images", iteration, index + 1, len(self.records))
            report = {"iteration": iteration, "images": len(self.records),
                      "full_training_set": self.settings.MAX_IMAGES == 0,
                      "view": "fixed clean resized train images; native segmentation feature cells",
                      "mask_repeats": self.settings.MASK_REPEATS,
                      "masked_prediction": "probability mean over masks hiding each cell; repeat survival also scored",
                      "rebuilder_has_passed_warmup": iteration > self.config.SRMP.WARMUP_ITERS,
                      "read_only": True, "direction1": probe.summary(), "objects": None,
                      "settings": dict(self.settings)}
            if self.settings.DINO:
                stats = {key: torch.cat(values) for key, values in all_stats.items()}
                previous, prior_iteration = self._previous(iteration)
                report["objects"] = object_summary(stats, self.graph, self.settings.MIN_SIMILARITY,
                                                    self.settings.GT_REGION_PURITY, previous)
                report["objects"].update({"reference": self.reference_meta, "fingerprint": self.fingerprint,
                                           "previous_iteration": prior_iteration,
                                           "proposal_patch_coverage": divide(retained_patches, total_patches),
                                           "interpretation": "feature-coherent proposals and similar cross-image regions; not physical instance identity"})
                save_state({"fingerprint": self.fingerprint, "iteration": iteration,
                            "input_size": list(self.settings.INPUT_SIZE),
                            "label": stats["label"].long(), "probability": stats["probability"].float()},
                           self.output / "regions_iter{:06d}.pth".format(iteration))
                self._write_csv("objects_iter{:06d}.csv".format(iteration), region_rows)
                self._write_pairs(iteration, stats)
            report["seconds"] = time.time() - started
            self._write_csv("errors_iter{:06d}.csv".format(iteration), probe.image_rows)
            save_json(report, self.output / "audit_iter{:06d}.json".format(iteration))
            self._timeline(report)
            self.logger.info("SRMP read-only diagnostics saved at %s (%.1fs)", self.output, report["seconds"])

    def _write_csv(self, filename, rows):
        if not rows:
            return
        path = self.output / filename
        with open(str(path) + ".tmp", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(path) + ".tmp", path)

    def _write_pairs(self, iteration, stats):
        rows = []
        for query in range(len(self.graph["index"])):
            for other, similarity in zip(self.graph["index"][query].tolist(), self.graph["similarity"][query].tolist()):
                if other < 0 or similarity < self.settings.MIN_SIMILARITY:
                    continue
                rows.append({"image": self.records[int(self.region_image_ids[query])]["name"],
                             "region_id": int(self.region_local_ids[query]),
                             "neighbour_image": self.records[int(self.region_image_ids[other])]["name"],
                             "neighbour_region_id": int(self.region_local_ids[other]), "dino_cosine": similarity,
                             "teacher_label": int(stats["label"][query]), "neighbour_teacher_label": int(stats["label"][other]),
                             "teacher_agrees": int(stats["label"][query] == stats["label"][other]),
                             "gt_label": int(stats["gt_label"][query]), "neighbour_gt_label": int(stats["gt_label"][other]),
                             "gt_purity": float(stats["gt_purity"][query]), "neighbour_gt_purity": float(stats["gt_purity"][other])})
        self._write_csv("pairs_iter{:06d}.csv".format(iteration), rows)

    def _timeline(self, report):
        agreement = report["direction1"]["agreement"]["teacher_masked"]
        scores = report["direction1"]["score_metrics"]
        row = {"iteration": report["iteration"], "images": report["images"], "seconds": report["seconds"],
               "error_given_agree": agreement["p_error_given_agreement"],
               "error_given_disagree": agreement["p_error_given_disagreement"],
               "disagreement_error_recall": agreement["error_recall_of_disagreement"]}
        for name in ("teacher_masked_js", "student_masked_js", "one_minus_confidence", "teacher_entropy"):
            for subset in ("all", "high_confidence"):
                row[subset + "_" + name + "_auroc"] = scores[subset + "/" + name]["error_detection_auroc_binned"]
        for name in ("within_region_majority_purity_macro", "teacher_cross_image_agreement", "random_cross_image_agreement",
                     "temporal_label_flip_rate", "temporal_probability_js", "proposal_patch_coverage", "teacher_class_entropy"):
            row[name] = report["objects"].get(name) if report["objects"] else None
        path = self.output / "metrics.csv"
        previous = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                previous = [r for r in csv.DictReader(handle) if int(r["iteration"]) != report["iteration"]]
        self._write_csv("metrics.csv", sorted(previous + [row], key=lambda r: int(r["iteration"])))
