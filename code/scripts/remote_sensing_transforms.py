"""Paired remote-sensing transforms shared by source and target training.

The ``rs_standard`` policy follows the public SiamSeg/MMsegmentation data
recipe in spirit: random scale, class-aware crop on labeled source images,
horizontal flip and photometric distortion. It is deliberately named
differently because this implementation is not byte-for-byte MMsegmentation.
"""
# SiamSeg/MMsegmentation ports below: Copyright 2020 The MMSegmentation Authors.
# Modified for PIL inputs and explicit IRRG plane handling. Apache-2.0;
# see references/siamseg/LICENSE for the upstream license.

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance


def siamseg_photometric(image: Image.Image) -> Image.Image:
    """OpenMMLab PhotoMetricDistortion, as vendored in SiamSeg (Apache-2.0).

    Match uint8 BGR/HSV arithmetic and NumPy RNG, then return stored RGB planes
    (IRRG for Vaihingen). No physical conversion of NIR into visible blue.
    """
    import cv2
    rng = np.random
    img = np.asarray(image)[:, :, ::-1].copy()

    def convert(array, alpha=1., beta=0.):
        return np.clip(array.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    def contrast(array):
        return convert(array, alpha=rng.uniform(.5, 1.5)) if rng.randint(2) else array

    if rng.randint(2):
        img = convert(img, beta=rng.uniform(-32, 32))
    mode = rng.randint(2)
    if mode == 1:
        img = contrast(img)
    if rng.randint(2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 1] = convert(img[:, :, 1], alpha=rng.uniform(.5, 1.5))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    if rng.randint(2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 0] = (img[:, :, 0].astype(int) + rng.randint(-18, 18)) % 180
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    if mode == 0:
        img = contrast(img)
    return Image.fromarray(img[:, :, ::-1].copy())


def siamseg_isprs_geometry(image, label, crop):
    """SiamSeg VH->PD square-chip Resize/RandomCrop/RandomFlip recipe."""
    import cv2
    if image.size != (crop, crop) or label is None:
        raise ValueError("siamseg_isprs requires labeled square chips matching crop size")
    size = int(crop * np.random.uniform(1., 2.))
    img = cv2.resize(np.asarray(image), (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(np.asarray(label), (size, size), interpolation=cv2.INTER_NEAREST)

    def position():
        return np.random.randint(0, size-crop+1), np.random.randint(0, size-crop+1)

    y, x = position()
    for _ in range(10):
        values, counts = np.unique(mask[y:y+crop, x:x+crop], return_counts=True)
        counts = counts[values != 255]
        if len(counts) > 1 and counts.max() / counts.sum() < .75:
            break
        y, x = position()
    img, mask = img[y:y+crop, x:x+crop], mask[y:y+crop, x:x+crop]
    if np.random.rand() < .5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    return Image.fromarray(img.copy()), Image.fromarray(mask.copy()), Image.new("L", (crop, crop), 255)


def siamseg_sfuda_geometry(image, label, crop):
    """SiamSeg geometry with a target crop that never inspects ground truth."""
    import cv2
    if image.size != (crop, crop):
        raise ValueError('SiamSeg SFUDA expects square native patches')
    size = int(crop * np.random.uniform(1., 2.))
    img = cv2.resize(np.asarray(image), (size, size), interpolation=cv2.INTER_LINEAR)
    mask = (cv2.resize(np.asarray(label), (size, size), interpolation=cv2.INTER_NEAREST)
            if label is not None else None)
    def position():
        return np.random.randint(0, size-crop+1), np.random.randint(0, size-crop+1)
    y, x = position()
    if mask is not None:
        for _ in range(10):
            values, counts = np.unique(mask[y:y+crop, x:x+crop], return_counts=True)
            counts = counts[values != 255]
            if len(counts) > 1 and counts.max() / counts.sum() < .75:
                break
            y, x = position()
    img = img[y:y+crop, x:x+crop]
    if mask is not None:
        mask = mask[y:y+crop, x:x+crop]
    if np.random.rand() < .5:
        img = img[:, ::-1]
        if mask is not None:
            mask = mask[:, ::-1]
    return Image.fromarray(img.copy()), (Image.fromarray(mask.copy()) if mask is not None else None), Image.new('L', (crop, crop), 255)


def photometric(image: Image.Image, rng=random) -> Image.Image:
    if rng.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.5:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.5:
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.8, 1.2))
    return image


def train_transform(image: Image.Image, label: Image.Image | None, crop: int,
                    policy: str, rng=random, scale_range=None):
    """Return aligned image, label and validity mask; never infer target labels."""
    if policy == "siamseg_isprs":
        return siamseg_isprs_geometry(image, label, crop)
    if policy == "siamseg_sfuda":
        return siamseg_sfuda_geometry(image, label, crop)
    if policy not in ("basic", "rs_standard"):
        raise ValueError(f"Unknown training policy: {policy}")
    if policy == "rs_standard":
        # ISPRS manifests already contain 512 px chips; LoveDA keeps 1024 px
        # originals. Avoid filling an ordinary ISPRS minibatch with padding.
        low, high = scale_range or (min(1.5, max(0.5, crop / min(image.size))), 1.5)
        scale = rng.uniform(low, high)
        size = tuple(max(1, round(side * scale)) for side in image.size)
        image = image.resize(size, Image.BILINEAR)
        if label is not None:
            label = label.resize(size, Image.NEAREST)
    width, height = image.size
    if policy == "basic" and min(width, height) < crop:
        raise ValueError(f"Image smaller than crop size {crop}: {image.size}")
    choices = 10 if policy == "rs_standard" and label is not None else 1
    for _ in range(choices):
        x = rng.randint(0, max(0, width - crop))
        y = rng.randint(0, max(0, height - crop))
        box = (x, y, min(x + crop, width), min(y + crop, height))
        chosen_label = label.crop(box) if label is not None else None
        if choices == 1 or chosen_label is None:
            break
        values, counts = np.unique(np.asarray(chosen_label), return_counts=True)
        counts = counts[values != 255]
        if len(counts) > 1 and counts.max() / counts.sum() < 0.75:
            break
    image = image.crop(box)
    label = chosen_label
    valid = Image.new("L", image.size, 255)
    if image.size != (crop, crop):
        padded = Image.new("RGB", (crop, crop), (123, 116, 104))
        padded.paste(image, (0, 0))
        image = padded
        if label is not None:
            padded_label = Image.new("L", (crop, crop), 255)
            padded_label.paste(label, (0, 0))
            label = padded_label
        padded_valid = Image.new("L", (crop, crop), 0)
        padded_valid.paste(valid, (0, 0))
        valid = padded_valid
    if rng.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        valid = valid.transpose(Image.FLIP_LEFT_RIGHT)
        if label is not None:
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return image, label, valid
