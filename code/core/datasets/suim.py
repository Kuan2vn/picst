import os
import numpy as np
import random
import torch
from torch.utils import data
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from os import listdir
from os.path import splitext

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

SUIM_COLOR_TO_TRAIN_ID = {
    (0, 0, 0): 0,          # background / water
    (0, 0, 255): 1,        # human divers
    (0, 255, 0): 2,        # aquatic plants / sea-grass
    (0, 255, 255): 3,      # wrecks / ruins
    (255, 0, 0): 4,        # robots / instruments
    (255, 0, 255): 5,      # reefs / invertebrates
    (255, 255, 0): 6,      # fish / vertebrates
    (255, 255, 255): 7,    # sea-floor / rocks
}

UIIS_COLOR_TO_TRAIN_ID = {
    (0, 0, 0): 0,
    (0, 0, 255): 6,        # fish
    (0, 255, 0): 5,        # reefs
    (255, 0, 0): 2,        # aquatic plants
    (255, 255, 0): 3,      # wrecks / ruins
    (255, 0, 255): 1,      # human divers
    (0, 255, 255): 4,      # robots
    (255, 128, 0): 7,      # sea-floor
}

UIIS_INDEX_TO_TRAIN_ID = {
    0: 0,
    1: 6,
    2: 5,
    3: 2,
    4: 3,
    5: 1,
    6: 4,
    7: 7,
}

LEGACY_SUIM_COLOR_TO_TRAIN_ID = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,
    (0, 255, 0): 2,
    (255, 0, 0): 3,
    (255, 255, 0): 4,
    (255, 0, 255): 5,
    (0, 255, 255): 6,
}

UNDERWATER_TRAIN_ID_TO_NAME = {
    0: "Background",
    1: "Human divers",
    2: "Aquatic plants/sea-grass",
    3: "Wrecks/ruins",
    4: "Robots/instruments",
    5: "Reefs and invertebrates",
    6: "Fish and vertebrates",
    7: "Sea-floor and rocks",
}


def _cfg_data_flag(cfg, name, default=False):
    if cfg is None or not hasattr(cfg, "DATASETS"):
        return default
    return getattr(cfg.DATASETS, name) if name in cfg.DATASETS else default


def _list_image_files(imgs_dir):
    return sorted(
        file
        for file in listdir(imgs_dir)
        if not file.startswith(".") and splitext(file)[1].lower() in IMAGE_EXTENSIONS
    )


def _normalize_suffixes(mask_suffixes):
    if mask_suffixes is None:
        return (".bmp", ".png")
    if isinstance(mask_suffixes, str):
        return (mask_suffixes,)
    return tuple(mask_suffixes)


def _label_path(masks_dir, name, mask_suffixes):
    for suffix in mask_suffixes:
        path = os.path.join(masks_dir, name + suffix)
        if os.path.exists(path):
            return path
    return os.path.join(masks_dir, name + mask_suffixes[0])


def _color_map_for_palette(label_palette, legacy_color_map):
    if legacy_color_map:
        return LEGACY_SUIM_COLOR_TO_TRAIN_ID.copy(), None
    if label_palette == "uiis":
        return UIIS_COLOR_TO_TRAIN_ID.copy(), UIIS_INDEX_TO_TRAIN_ID.copy()
    return SUIM_COLOR_TO_TRAIN_ID.copy(), None


def _trainid2name(num_classes):
    return {
        train_id: name
        for train_id, name in UNDERWATER_TRAIN_ID_TO_NAME.items()
        if train_id < num_classes
    }


class SuimDataSet(data.Dataset):
    def __init__(
            self,
            imgs_dir,
            masks_dir,
            max_iters=None,
            num_classes=7,
            split="train",
            transform=None,
            cfg=None,
            ignore_label=255,
            debug=False,
            pseudo=False,
            label_palette="suim",
            mask_suffixes=(".bmp", ".png"),
    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_list = []
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.mask_suffixes = _normalize_suffixes(mask_suffixes)
        self.label_palette = label_palette
        self.seafloor_ignore = bool(
            _cfg_data_flag(cfg, "SEAFLOOR_IGNORE", False) or num_classes <= 7
        )
        self.legacy_color_map = bool(_cfg_data_flag(cfg, "SUIM_LEGACY_COLOR_MAP", False))
        self.color2id, self.index2id = _color_map_for_palette(
            self.label_palette, self.legacy_color_map
        )
        self.trainid2name = _trainid2name(num_classes)
        self.ids = [splitext(file)[0] for file in _list_image_files(imgs_dir)]

        for image_file in _list_image_files(imgs_dir):
            name = splitext(image_file)[0].strip()
            if pseudo:
                self.data_list.append(
                    {
                        "img": os.path.join(
                            self.imgs_dir, image_file
                        ),
                        "label": os.path.join(
                            cfg.OUTPUT_DIR, "CTR_O/%s"
                            % (
                                name + '.png',
                            ),
                        ),
                        "name": name,
                    }
                )
            else:
                self.data_list.append(
                    {
                        "img": os.path.join(
                            self.imgs_dir, image_file
                        ),
                        "label": _label_path(self.masks_dir, name, self.mask_suffixes),
                        "name": name,
                    }
                )

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        self.transform = transform
        self.ignore_label = ignore_label
        self.debug = debug
        self.pseudo = pseudo

    def __len__(self):
        return len(self.data_list)

    def rgb_to_label(self, img):
        label = self.ignore_label * np.ones(img.shape[:2], dtype=np.float32)
        for color, idx in self.color2id.items():
            mask = np.all(img == color, axis=-1)
            label[mask] = idx
        return self._finalize_label(label)

    def index_to_label(self, arr):
        label = self.ignore_label * np.ones(arr.shape[:2], dtype=np.float32)
        if self.index2id is None:
            valid = (arr >= 0) & (arr < 255)
            label[valid] = arr[valid]
        else:
            for raw_id, train_id in self.index2id.items():
                label[arr == raw_id] = train_id
        return self._finalize_label(label)

    def _finalize_label(self, label):
        if self.seafloor_ignore:
            label[label == 7] = self.ignore_label
        label[(label < 0) | ((label >= self.NUM_CLASS) & (label != self.ignore_label))] = self.ignore_label
        return label

    def mask_to_label(self, mask_img):
        if mask_img.mode in ("L", "P", "I", "I;16"):
            return self.index_to_label(np.array(mask_img))
        return self.rgb_to_label(np.array(mask_img.convert('RGB')))

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"] + '.png'

        if not self.pseudo:
            # SUIM original masks are RGB
            label_arr = self.mask_to_label(Image.open(datafiles["label"]))
            label = Image.fromarray(label_arr)
        else:
            # Pseudo labels are saved as integer masks
            label = Image.open(datafiles["label"])
            label_arr = self.index_to_label(np.array(label))
            label = Image.fromarray(label_arr)

        if self.transform is not None:
            image, label, _ = self.transform(image, label)
            
        return image, label, name

from .transform import Compose
from copy import deepcopy
import cv2

def denormalizeimage(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    images = images.cpu().numpy()
    images = images.transpose((0,2,3,1))
    images *= std
    images += mean
    images *= 255.0
    images = images.transpose((0,3,1,2))
    return torch.tensor(images)

from skimage.measure import label as sklabel
import pickle

class rand_mixer_suim():
    def __init__(self, cfg, root='data/SUIM/orin_data/train_val/images/'):
        self.image_root = root
        self.cdd_root = os.path.join(cfg.OUTPUT_DIR)
        self.class_num = cfg.MODEL.NUM_CLASSES
        candidate_name = 'CTR'
        self.candidate_name = candidate_name
        self.file_list = os.path.join(cfg.OUTPUT_DIR, candidate_name + '.p')
        self.label_to_file, _ = pickle.load(open(self.file_list, "rb"))
        self.join_mode = 'direct'
        
    def mix(self, in_imgs, classes, choice_p, for_show=False):
        in_imgs = np.array(in_imgs)
        out_imgs = deepcopy(in_imgs)
        out_lbls = np.ones(in_imgs.shape[:2]) * 255 
        in_w, in_h = in_imgs.shape[:2]
        if hasattr(choice_p, 'cpu'):
            choice_p = choice_p.cpu().numpy()
        choice_p = np.array(choice_p).astype('float64')
        choice_p=choice_p/choice_p.sum()

        valid_indices = [i for i, c in enumerate(classes) if len(self.label_to_file.get(c, [])) > 0]
        if len(valid_indices) == 0:
            return out_imgs, out_lbls
        valid_classes = [classes[i] for i in valid_indices]
        valid_p = np.array([choice_p[i] for i in valid_indices])
        if valid_p.sum() == 0:
            valid_p = np.ones(len(valid_classes)) / len(valid_classes)
        else:
            valid_p = valid_p / valid_p.sum()
            
        class_idx = np.random.choice(valid_classes, size=1, p=valid_p)[0]
        ins_pix_num = 50
        while True:
            name = random.sample(self.label_to_file[class_idx], 1)
            cdd_img_path = os.path.join(self.image_root, "%s" % name[0].split('.')[0] + '.jpg')
            cdd_label_path = os.path.join(self.cdd_root, "{}/{}".format(self.candidate_name, name[0].split('.')[0] + '.png'))
            
            try:
                cdd_img = Image.open(cdd_img_path).convert('RGB')
                cdd_lbl = Image.open(cdd_label_path)
            except:
                continue
                
            cdd_lbl = cdd_lbl.resize(cdd_img.size)
            
            cdd_img = np.asarray(cdd_img, np.uint8)
            cdd_lbl = np.asarray(cdd_lbl, np.uint8)

            mask = cdd_lbl==class_idx           
            if np.sum(mask)>=ins_pix_num:
                break
        
        masknp = mask.astype(int) 
        seg, forenum = sklabel(masknp, background=0, return_num=True, connectivity=2)
        filled_mask = np.zeros(in_imgs.shape[:2])
        filled_boxes = []
        for i in range(forenum):
            instance_id = i+1
            if np.sum(seg==instance_id) < 20:
                continue
            ins_mask = (seg==instance_id).astype(np.uint8)
            cont, hierarchy = cv2.findContours(ins_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
            cont = sorted(cont, key=lambda c: cv2.contourArea(c), reverse=True)                    
            x, y, w, h = cv2.boundingRect(cont[0])   
            
            randscale = 1.0 + np.random.rand() * 0.5
            resized_crop_ins = cv2.resize(cdd_img[y:y+h, x:x+w], None, fx=randscale, fy=randscale, interpolation = cv2.INTER_NEAREST)
            resized_crop_mask = cv2.resize(ins_mask[y:y+h, x:x+w], None, fx=randscale, fy=randscale, interpolation = cv2.INTER_NEAREST)
            resized_crop_mask_cdd = cv2.resize(cdd_lbl[y:y+h, x:x+w], None, fx=randscale, fy=randscale, interpolation = cv2.INTER_NEAREST)
            new_w, new_h = resized_crop_ins.shape[:2]
            if in_w <= new_w or in_h <= new_h:
                continue
            
            cnt = 100
            while cnt>1:
                if class_idx not in resized_crop_mask_cdd[resized_crop_mask>0] and len(filled_boxes) > 0:
                    rand_box = random.sample(filled_boxes, 1)[0]
                    x1 = random.randint(rand_box[0], rand_box[1] - 1)
                    y1 = random.randint(rand_box[2], rand_box[3] - 1)
                    if x1+new_w<in_w-1 and y1+new_h<in_h-1:
                        break
                x1 = random.randint(0, in_w - new_w - 1)
                y1 = random.randint(0, in_h - new_h - 1)
                if filled_mask[x1, y1] == 0 and filled_mask[x1+new_w, y1+new_h] == 0:
                    break
                cnt-=1
            
            if cnt>1:
                out_imgs[x1:x1+new_w, y1:y1+new_h][resized_crop_mask>0] = resized_crop_ins[resized_crop_mask>0]
                out_lbls[x1:x1+new_w, y1:y1+new_h][resized_crop_mask>0] = resized_crop_mask_cdd[resized_crop_mask>0]
                filled_mask[x1:x1+new_w, y1:y1+new_h] = 1 
                
                if len(np.unique(resized_crop_mask_cdd[resized_crop_mask>0])) == 1 and class_idx in resized_crop_mask_cdd[resized_crop_mask>0]:
                    filled_boxes.append([x1, x1+new_w, y1,y1+new_h])
                    
        return out_imgs, out_lbls

class SuimDataSetTrain(data.Dataset):
    def __init__(
            self,
            imgs_dir,
            masks_dir,
            max_iters=None,
            num_classes=7,
            split="train",
            transform=None,
            rsc=None,
            cfg=None,
            ignore_label=255,
            debug=False,
            pseudo=False,
            label_palette="suim",
            mask_suffixes=(".bmp", ".png"),
    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_list = []
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.mask_suffixes = _normalize_suffixes(mask_suffixes)
        self.label_palette = label_palette
        self.seafloor_ignore = bool(
            _cfg_data_flag(cfg, "SEAFLOOR_IGNORE", False) or num_classes <= 7
        )
        self.legacy_color_map = bool(_cfg_data_flag(cfg, "SUIM_LEGACY_COLOR_MAP", False))
        self.color2id, self.index2id = _color_map_for_palette(
            self.label_palette, self.legacy_color_map
        )
        self.trainid2name = _trainid2name(num_classes)
        image_files = _list_image_files(imgs_dir)
        self.ids = [splitext(file)[0] for file in image_files]

        self.img_ids = self.ids

        for image_file in image_files:
            name = splitext(image_file)[0]
            self.data_list.append(
                {
                    "img": os.path.join(
                        self.imgs_dir, image_file
                    ),
                    "label": _label_path(self.masks_dir, name, self.mask_suffixes),
                    "name": name,
                }
            )

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        self.mixup = cfg.TCR.OPEN if cfg is not None and rsc is not None else False
        if self.mixup:
            self.mixer = rand_mixer_suim(cfg)
            self.mix_classes = list(range(1, self.NUM_CLASS))
            self.mix_p = np.ones(len(self.mix_classes)) / len(self.mix_classes)

        self.transform = transform
        self.ignore_label = ignore_label
        self.debug = debug
        self.pseudo = pseudo

    def __len__(self):
        return len(self.data_list)

    def rgb_to_label(self, img):
        label = self.ignore_label * np.ones(img.shape[:2], dtype=np.float32)
        for color, idx in self.color2id.items():
            mask = np.all(img == color, axis=-1)
            label[mask] = idx
        return self._finalize_label(label)

    def index_to_label(self, arr):
        label = self.ignore_label * np.ones(arr.shape[:2], dtype=np.float32)
        if self.index2id is None:
            valid = (arr >= 0) & (arr < 255)
            label[valid] = arr[valid]
        else:
            for raw_id, train_id in self.index2id.items():
                label[arr == raw_id] = train_id
        return self._finalize_label(label)

    def _finalize_label(self, label):
        if self.seafloor_ignore:
            label[label == 7] = self.ignore_label
        label[(label < 0) | ((label >= self.NUM_CLASS) & (label != self.ignore_label))] = self.ignore_label
        return label

    def mask_to_label(self, mask_img):
        if mask_img.mode in ("L", "P", "I", "I;16"):
            return self.index_to_label(np.array(mask_img))
        return self.rgb_to_label(np.array(mask_img.convert('RGB')))

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]
        
        self.img_size = (1280, 640)

        image = Image.open(datafiles["img"]).convert('RGB')
        name = datafiles["name"]

        if getattr(self, "ignore_real_labels", False):
            label = Image.fromarray(
                self.ignore_label * np.ones((image.size[1], image.size[0]), dtype=np.float32)
            )
        elif not self.pseudo:
            # SUIM original masks are RGB
            try:
                label_arr = self.mask_to_label(Image.open(datafiles["label"]))
            except:
                label_arr = np.zeros((image.size[1], image.size[0]), dtype=np.float32)
            label = Image.fromarray(label_arr)
        else:
            # Pseudo labels are saved as integer masks
            try:
                label_img = Image.open(datafiles["label"].replace('.bmp', '.png'))
                label_arr = self.index_to_label(np.array(label_img))
                label = Image.fromarray(label_arr)
            except:
                label = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.float32))

        ## full image
        image = image.resize(self.img_size, Image.BILINEAR)
        img_full = deepcopy(image)
        # full label
        label = label.resize(self.img_size, Image.NEAREST)     

        if self.transform is not None:
            trans_image, trans_label, trans_param = self.transform(image, deepcopy(label))
            
        # mixup 
        if self.mixup:
            trans_image = denormalizeimage(trans_image.unsqueeze(0))
            trans_image = np.asarray(trans_image[0].numpy(), dtype=np.uint8).transpose(1, 2, 0)
            trans_image, mix_label = self.mixer.mix(trans_image, self.mix_classes, self.mix_p)
            # norm
            trans_image = np.array(trans_image, dtype=np.float64) / 255
            trans_image -= [0.485, 0.456, 0.406]
            trans_image /= [0.229, 0.224, 0.225]
            trans_image = torch.from_numpy(trans_image.transpose(2, 0, 1)).float()
            mix_label = torch.from_numpy(mix_label)
        else:
            mix_label = torch.ones(trans_image.shape[-2:]) * 255
            
        # norm
        img_full = np.array(img_full, dtype=np.float64) / 255
        img_full -= [0.485, 0.456, 0.406]
        img_full /= [0.229, 0.224, 0.225]
        img_full = torch.from_numpy(img_full.transpose(2, 0, 1)).float()
        
        return trans_image, \
            torch.from_numpy(np.array(trans_label)),\
            name, \
            trans_param, \
            img_full, \
            mix_label
