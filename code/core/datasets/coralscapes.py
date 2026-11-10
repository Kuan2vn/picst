import os
import pickle
import random
from copy import deepcopy
from os import listdir
from os.path import splitext

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from skimage.measure import label as sklabel
from torch.utils import data

ImageFile.LOAD_TRUNCATED_IMAGES = True


CORALSCAPES_TRAINID2NAME = {
    0: "seagrass",
    1: "trash",
    2: "other coral dead",
    3: "other coral bleached",
    4: "sand",
    5: "other coral alive",
    6: "human",
    7: "transect tools",
    8: "fish",
    9: "algae covered substrate",
    10: "other animal",
    11: "unknown hard substrate",
    12: "background",
    13: "dark",
    14: "transect line",
    15: "massive/meandering bleached",
    16: "massive/meandering alive",
    17: "rubble",
    18: "branching bleached",
    19: "branching dead",
    20: "millepora",
    21: "branching alive",
    22: "massive/meandering dead",
    23: "clam",
    24: "acropora alive",
    25: "sea cucumber",
    26: "turbinaria",
    27: "table acropora alive",
    28: "sponge",
    29: "anemone",
    30: "pocillopora alive",
    31: "table acropora dead",
    32: "meandering bleached",
    33: "stylophora alive",
    34: "sea urchin",
    35: "meandering alive",
    36: "meandering dead",
    37: "crown of thorn",
    38: "dead clam",
}


def denormalizeimage(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    images = images.cpu().numpy()
    images = images.transpose((0, 2, 3, 1))
    images *= std
    images += mean
    images *= 255.0
    images = images.transpose((0, 3, 1, 2))
    return torch.tensor(images)


def image_stem_to_label_stem(stem):
    if stem.startswith("frame_"):
        return "mask_" + stem[len("frame_") :]
    return stem


def remap_coralscapes_label(label, ignore_label=255):
    label = np.asarray(label)
    label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
    valid = (label >= 1) & (label <= 39)
    label_copy[valid] = label[valid].astype(np.float32) - 1.0
    return label_copy


def resolve_full_size(cfg, default=(1024, 512)):
    if cfg is not None and hasattr(cfg, "INPUT") and "INPUT_SIZE_TEST" in cfg.INPUT:
        size = tuple(cfg.INPUT.INPUT_SIZE_TEST)
        if len(size) == 2:
            return size
    return default


class CoralScapesDataSet(data.Dataset):
    def __init__(
        self,
        imgs_dir,
        masks_dir,
        max_iters=None,
        num_classes=39,
        split="train",
        transform=None,
        cfg=None,
        ignore_label=255,
        debug=False,
        pseudo=False,
    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_list = []
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.ignore_label = ignore_label
        self.debug = debug
        self.pseudo = pseudo
        self.trainid2name = CORALSCAPES_TRAINID2NAME

        ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith(".")]
        ids = sorted(ids)
        for name in ids:
            if pseudo:
                label_path = os.path.join(cfg.OUTPUT_DIR, "CTR_O", name + ".png")
            else:
                label_name = image_stem_to_label_stem(name) + ".png"
                label_path = os.path.join(self.masks_dir, label_name)

            self.data_list.append(
                {
                    "img": os.path.join(self.imgs_dir, name + ".jpg"),
                    "label": label_path,
                    "name": name,
                }
            )

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert("RGB")
        name = datafiles["name"] + ".png"

        if not self.pseudo:
            label = np.array(Image.open(datafiles["label"]), dtype=np.float32)
            label = Image.fromarray(remap_coralscapes_label(label, self.ignore_label))
        else:
            label = Image.open(datafiles["label"])
            label = Image.fromarray(np.array(label, dtype=np.float32))

        if self.transform is not None:
            image, label, _ = self.transform(image, label)
        return image, label, name


class rand_mixer_coralscapes:
    def __init__(self, cfg, root="data/coralscapes/orin_data/train/images/"):
        self.image_root = root
        self.cdd_root = os.path.join(cfg.OUTPUT_DIR)
        self.class_num = cfg.MODEL.NUM_CLASSES
        self.candidate_name = "CTR"
        self.file_list = os.path.join(cfg.OUTPUT_DIR, self.candidate_name + ".p")
        self.label_to_file, _ = pickle.load(open(self.file_list, "rb"))
        self.join_mode = "direct"

    def mix(self, in_imgs, classes, choice_p, for_show=False):
        in_imgs = np.array(in_imgs)
        out_imgs = deepcopy(in_imgs)
        out_lbls = np.ones(in_imgs.shape[:2]) * 255
        in_h, in_w = in_imgs.shape[:2]
        if hasattr(choice_p, "cpu"):
            choice_p = choice_p.cpu().numpy()
        choice_p = np.array(choice_p).astype("float64")
        choice_p = choice_p / max(choice_p.sum(), 1e-12)

        valid_indices = [i for i, c in enumerate(classes) if len(self.label_to_file.get(c, [])) > 0]
        if len(valid_indices) == 0:
            return out_imgs, out_lbls
        valid_classes = [classes[i] for i in valid_indices]
        valid_p = np.array([choice_p[i] for i in valid_indices])
        valid_p = np.ones(len(valid_classes)) / len(valid_classes) if valid_p.sum() == 0 else valid_p / valid_p.sum()

        class_idx = np.random.choice(valid_classes, size=1, p=valid_p)[0]
        ins_pix_num = 50
        while True:
            name = random.sample(self.label_to_file[class_idx], 1)
            stem = name[0].split(".")[0]
            cdd_img_path = os.path.join(self.image_root, stem + ".jpg")
            cdd_label_path = os.path.join(self.cdd_root, self.candidate_name, stem + ".png")

            try:
                cdd_img = Image.open(cdd_img_path).convert("RGB")
                cdd_lbl = Image.open(cdd_label_path)
            except Exception:
                continue

            cdd_lbl = cdd_lbl.resize(cdd_img.size)
            cdd_img = np.asarray(cdd_img, np.uint8)
            cdd_lbl = np.asarray(cdd_lbl, np.uint8)

            mask = cdd_lbl == class_idx
            if np.sum(mask) >= ins_pix_num:
                break

        seg, forenum = sklabel(mask.astype(int), background=0, return_num=True, connectivity=2)
        filled_mask = np.zeros(in_imgs.shape[:2])
        filled_boxes = []
        for i in range(forenum):
            instance_id = i + 1
            if np.sum(seg == instance_id) < 20:
                continue
            ins_mask = (seg == instance_id).astype(np.uint8)
            cont, _ = cv2.findContours(ins_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if len(cont) == 0:
                continue
            cont = sorted(cont, key=lambda c: cv2.contourArea(c), reverse=True)
            x, y, w, h = cv2.boundingRect(cont[0])

            randscale = 1.0 + np.random.rand() * 0.5
            resized_crop_ins = cv2.resize(cdd_img[y : y + h, x : x + w], None, fx=randscale, fy=randscale, interpolation=cv2.INTER_NEAREST)
            resized_crop_mask = cv2.resize(ins_mask[y : y + h, x : x + w], None, fx=randscale, fy=randscale, interpolation=cv2.INTER_NEAREST)
            resized_crop_mask_cdd = cv2.resize(cdd_lbl[y : y + h, x : x + w], None, fx=randscale, fy=randscale, interpolation=cv2.INTER_NEAREST)
            new_h, new_w = resized_crop_ins.shape[:2]
            if in_h <= new_h or in_w <= new_w:
                continue

            cnt = 100
            while cnt > 1:
                if class_idx not in resized_crop_mask_cdd[resized_crop_mask > 0] and len(filled_boxes) > 0:
                    rand_box = random.sample(filled_boxes, 1)[0]
                    y1 = random.randint(rand_box[0], rand_box[1] - 1)
                    x1 = random.randint(rand_box[2], rand_box[3] - 1)
                    if y1 + new_h < in_h - 1 and x1 + new_w < in_w - 1:
                        break
                y1 = random.randint(0, in_h - new_h - 1)
                x1 = random.randint(0, in_w - new_w - 1)
                if filled_mask[y1, x1] == 0 and filled_mask[y1 + new_h, x1 + new_w] == 0:
                    break
                cnt -= 1

            if cnt > 1:
                out_imgs[y1 : y1 + new_h, x1 : x1 + new_w][resized_crop_mask > 0] = resized_crop_ins[resized_crop_mask > 0]
                out_lbls[y1 : y1 + new_h, x1 : x1 + new_w][resized_crop_mask > 0] = resized_crop_mask_cdd[resized_crop_mask > 0]
                filled_mask[y1 : y1 + new_h, x1 : x1 + new_w] = 1

                if len(np.unique(resized_crop_mask_cdd[resized_crop_mask > 0])) == 1 and class_idx in resized_crop_mask_cdd[resized_crop_mask > 0]:
                    filled_boxes.append([y1, y1 + new_h, x1, x1 + new_w])

        return out_imgs, out_lbls


class CoralScapesDataSetTrain(data.Dataset):
    def __init__(
        self,
        imgs_dir,
        masks_dir,
        max_iters=None,
        num_classes=39,
        split="train",
        transform=None,
        rsc=None,
        cfg=None,
        ignore_label=255,
        debug=False,
        pseudo=False,
    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_list = []
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.ignore_label = ignore_label
        self.debug = debug
        self.pseudo = pseudo
        self.trainid2name = CORALSCAPES_TRAINID2NAME
        self.img_size = resolve_full_size(cfg)

        self.ids = sorted(splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith("."))
        self.img_ids = self.ids

        for name in self.img_ids:
            self.data_list.append(
                {
                    "img": os.path.join(self.imgs_dir, name + ".jpg"),
                    "label": os.path.join(self.masks_dir, image_stem_to_label_stem(name) + ".png"),
                    "name": name,
                }
            )

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        self.mixup = cfg.TCR.OPEN if cfg is not None else False
        if self.mixup:
            self.mixer = rand_mixer_coralscapes(cfg, root=self.imgs_dir)
            self.mix_classes = list(range(num_classes))
            self.mix_p = np.ones(len(self.mix_classes)) / len(self.mix_classes)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert("RGB")
        name = datafiles["name"]

        if getattr(self, "ignore_real_labels", False):
            label = Image.fromarray(self.ignore_label * np.ones((image.size[1], image.size[0]), dtype=np.float32))
        elif not self.pseudo:
            try:
                raw_label = np.array(Image.open(datafiles["label"]), dtype=np.float32)
                label = Image.fromarray(remap_coralscapes_label(raw_label, self.ignore_label))
            except Exception:
                label = Image.fromarray(self.ignore_label * np.ones((image.size[1], image.size[0]), dtype=np.float32))
        else:
            try:
                label_img = Image.open(datafiles["label"].replace(".png", ".png"))
                label = Image.fromarray(np.array(label_img, dtype=np.float32))
            except Exception:
                label = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.float32))

        image = image.resize(self.img_size, Image.BILINEAR)
        img_full = deepcopy(image)
        label = label.resize(self.img_size, Image.NEAREST)

        if self.transform is not None:
            trans_image, trans_label, trans_param = self.transform(image, deepcopy(label))

        if self.mixup:
            trans_image = denormalizeimage(trans_image.unsqueeze(0))
            trans_image = np.asarray(trans_image[0].numpy(), dtype=np.uint8).transpose(1, 2, 0)
            trans_image, mix_label = self.mixer.mix(trans_image, self.mix_classes, self.mix_p)
            trans_image = np.array(trans_image, dtype=np.float64) / 255
            trans_image -= [0.485, 0.456, 0.406]
            trans_image /= [0.229, 0.224, 0.225]
            trans_image = torch.from_numpy(trans_image.transpose(2, 0, 1)).float()
            mix_label = torch.from_numpy(mix_label)
        else:
            mix_label = torch.ones(trans_image.shape[-2:]) * 255

        img_full = np.array(img_full, dtype=np.float64) / 255
        img_full -= [0.485, 0.456, 0.406]
        img_full /= [0.229, 0.224, 0.225]
        img_full = torch.from_numpy(img_full.transpose(2, 0, 1)).float()

        return trans_image, torch.from_numpy(np.array(trans_label)), name, trans_param, img_full, mix_label
