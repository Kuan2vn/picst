"""DT-ST negative learning and label-free teacher feedback for the B2 adapter."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def validate_settings(settings):
    if type(settings["dynamic_teacher"]) is not bool:
        raise ValueError("dynamic_teacher must be boolean")
    for key in ("query_start", "query_step", "max_interval", "points", "fixed_interval"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"Invalid {key}")
    if (settings["query_start"] > settings["max_interval"] or
            not 0 < settings["ema"] < 1 or not math.isfinite(settings["negative_weight"]) or
            settings["negative_weight"] < 0):
        raise ValueError("Invalid teacher/loss settings")
    # Optional Stable-ST proxy (baseline overlay 08/11/2026); absent -> the released DT-ST soft-ND density.
    if settings.get("proxy", "soft_nd") not in ("soft_nd", "nfd"):
        raise ValueError("Unknown teacher-feedback proxy")
    return settings


def negative_loss(logits, teacher_probability, valid, weight=.5, protected_labels=None):
    # Match released DT-ST sigmoid BCE; geometry padding is not supervision.
    selected = (teacher_probability.detach() <= 1 / logits.shape[1]) & valid[:, None].bool()
    values = logits[selected]
    if not values.numel():
        return logits.sum() * 0
    if protected_labels is None:
        return weight * F.binary_cross_entropy_with_logits(values, torch.zeros_like(values))
    # A recovered positive class must not receive contradictory negative supervision.
    protected = torch.arange(logits.shape[1],device=logits.device)[None,:,None,None] == protected_labels[:,None]
    terms = F.binary_cross_entropy_with_logits(values, torch.zeros_like(values), reduction='none')
    return weight * (terms * ~protected[selected]).mean()


@torch.no_grad()
def probe_probability(encoder, classifier, images):
    modes = [(module, module.training) for model in (encoder, classifier) for module in model.modules()]
    try:
        encoder.eval(); classifier.eval()
        return classifier(encoder(images)[1], images.shape[-2:]).softmax(1)
    finally:
        # Restore individual modes, including the frozen source decoder BN.
        for module, mode in modes:
            module.training = mode


def density(probability, indices):
    values = probability.permute(0, 2, 3, 1).reshape(-1, probability.shape[1])
    values = F.normalize(values[indices.to(values.device)], dim=1)
    affinity = (values @ values.T * 20).softmax(1)
    return float(-(affinity * (affinity + 1e-5).log()).sum(1).mean())


class TeacherFeedback:
    def __new__(cls, settings, device):
        if cls is TeacherFeedback and settings.get("proxy", "soft_nd") == "nfd":
            return NFDFeedback(settings, device)
        return super().__new__(cls)

    def __init__(self, settings, device):
        self.settings, self.device = settings, device
        self.buffer = []
        self.probed_images = self.query_images = self.queries = self.updates = 0

    @torch.no_grad()
    def observe(self, encoder, classifier, images):
        probability = probe_probability(encoder, classifier, images)
        count = probability.shape[0] * probability.shape[2] * probability.shape[3]
        indices = torch.randperm(count)[:self.settings["points"]]
        score = density(probability, indices)
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite DT-ST feedback")
        self.buffer.append((images.detach().cpu().clone(), indices, score))
        self.probed_images += len(images)

    @torch.no_grad()
    def query(self, encoder, classifier):
        size = len(self.buffer)
        if size < self.settings["query_start"] or (size-self.settings["query_start"]) % self.settings["query_step"]:
            return None
        changes = []
        for images, indices, old_score in self.buffer:
            now = density(probe_probability(encoder, classifier, images.to(self.device)), indices)
            if not math.isfinite(now):
                raise FloatingPointError("Non-finite DT-ST query")
            changes.append(now > old_score)
            self.query_images += len(images)
        self.queries += 1
        gain = sum(changes) / len(changes)
        forced = size > self.settings["max_interval"]
        update = gain > .5 or forced
        event = {"interval": size, "gain_rate": gain, "forced": forced, "update": update}
        if update:
            self.updates += 1
            self.buffer.clear()
        return event

    def state_dict(self):
        return {name: getattr(self, name) for name in ("buffer", "probed_images", "query_images", "queries", "updates")}

    def load_state_dict(self, state):
        for name in ("buffer", "probed_images", "query_images", "queries", "updates"):
            setattr(self, name, state[name])

    def metrics(self):
        return {"dtst_probe_images": self.probed_images, "dtst_query_images": self.query_images,
                "dtst_queries": self.queries, "dtst_teacher_updates": self.updates,
                "dtst_buffer_batches": len(self.buffer)}


@torch.no_grad()
def _low_freq_amp(x, keep_ratio=0.125):
    # Stable-ST Q_style: grey-level low-frequency FFT amplitude (author core/apis + train_stable_st.py).
    amp = torch.fft.rfft2(x.mean(dim=1, keepdim=True).float(), norm="ortho").abs()
    h, w2 = amp.shape[-2], amp.shape[-1]
    return amp[:, :, :max(1, int(h * keep_ratio)), :max(1, int(w2 * keep_ratio))].flatten(1)


@torch.no_grad()
def nfd_scores(encoder, queries, neighbours):
    """Neighbour Feature Density (Stable-ST, IJCV 2026): 1 - mean entropy of softmax_{hw} <F_q, F_n>.
    The author applies it to the DeepLab layer4 tensor; the B2 analogue is the last MiT stage (encoder(x)[0])."""
    modes = [(module, module.training) for module in encoder.modules()]
    try:
        encoder.eval()
        fq, fn = encoder(queries)[0].float(), encoder(neighbours)[0].float()
    finally:
        for module, mode in modes:
            module.training = mode
    s = (fq * fn).sum(dim=1).flatten(1)
    p = s.softmax(1)
    return (1.0 - (-(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).mean(dim=1))).tolist()


class NFDFeedback:
    """Stable-ST dynamic teacher feedback. Buffer length counts images (the author extends by batch size);
    query when length >= query_start and (length - query_start) % query_step == 0; update if more than half of
    the stored queries improved their NFD, or when length > max_interval."""

    def __init__(self, settings, device):
        self.settings, self.device = settings, device
        self.buffer = []  # (query images, neighbour images, NFD scores)
        self.probed_images = self.query_images = self.queries = self.updates = 0

    def _length(self):
        return sum(len(scores) for _, _, scores in self.buffer)

    @torch.no_grad()
    def observe(self, encoder, classifier, images):
        probability = probe_probability(encoder, classifier, images)
        entropy_score = (-(probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log()).sum(1)).mean(dim=(1, 2))
        style = F.normalize(_low_freq_amp(images), dim=1)
        pool = torch.argsort(entropy_score)[:max(1, len(images) // 2)]
        neighbours = []
        for i in range(len(images)):
            similarity = style[i:i + 1] @ style[pool].T
            # Intended self-exclusion; the released line passes an index tensor to masked_fill.
            similarity = similarity[0].masked_fill(pool == i, -1.0)
            neighbours.append(int(pool[similarity.argmax()]))
        neighbour_images = images[neighbours]
        scores = nfd_scores(encoder, images, neighbour_images)
        if not all(math.isfinite(v) for v in scores):
            raise FloatingPointError("Non-finite Stable-ST NFD feedback")
        self.buffer.append((images.detach().cpu().clone(), neighbour_images.detach().cpu().clone(), scores))
        self.probed_images += len(images)

    @torch.no_grad()
    def query(self, encoder, classifier):
        size = self._length()
        if size < self.settings["query_start"] or (size - self.settings["query_start"]) % self.settings["query_step"]:
            return None
        changes = []
        for queries, neighbours, old in self.buffer:
            now = nfd_scores(encoder, queries.to(self.device), neighbours.to(self.device))
            if not all(math.isfinite(v) for v in now):
                raise FloatingPointError("Non-finite Stable-ST NFD query")
            changes += [a > b for a, b in zip(now, old)]
            self.query_images += len(queries)
        self.queries += 1
        gain = sum(changes) / len(changes)
        forced = size > self.settings["max_interval"]
        update = gain > .5 or forced
        event = {"interval": size, "gain_rate": gain, "forced": forced, "update": update, "proxy": "nfd"}
        if update:
            self.updates += 1
            self.buffer.clear()
        return event

    def state_dict(self):
        return {name: getattr(self, name) for name in ("buffer", "probed_images", "query_images", "queries", "updates")}

    def load_state_dict(self, state):
        for name in ("buffer", "probed_images", "query_images", "queries", "updates"):
            setattr(self, name, state[name])

    def metrics(self):
        return {"dtst_probe_images": self.probed_images, "dtst_query_images": self.query_images,
                "dtst_queries": self.queries, "dtst_teacher_updates": self.updates,
                "dtst_buffer_batches": len(self.buffer)}
