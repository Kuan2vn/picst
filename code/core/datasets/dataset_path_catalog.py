import os
from .cityscapes_b import cityscapesDataSetTrain 
from .cityscapes_meta import cityscapesDataSetMeta
from .cityscapes import cityscapesDataSet
from .cityscapes_self_distill import cityscapesSelfDistillDataSet
from .synthia import synthiaDataSet
from .gta5 import GTA5DataSet
from .bdd import BddDataSet
from .bdd_b import BddDataSetTrain 
from .bdd_meta import BddDataSetMeta
from .sd_llama70b import SDllama70bDataSet
from .suim import SuimDataSet, SuimDataSetTrain
from .coralscapes import CoralScapesDataSet, CoralScapesDataSetTrain


class DatasetCatalog(object):
    DATASET_DIR = "data"
    DATASETS = {
        "gta5_train": {
            "data_dir": "gta5",
            "data_list": "gta5_train_list.txt"
        },
        "gta5_val": {
            "data_dir": "gta5",
            "data_list": "gta5_train_list.txt"
        },
        "synthia_train": {
            "data_dir": "synthia",
            "data_list": "synthia_train_list.txt"
        },
        "cityscapes_train": {
            "data_dir": "cityscape",
            "data_list": "cityscapes_train_list.txt"
        },
        "cityscapes_TT_train": {
            "data_dir": "cityscape",
            "data_list": "cityscapes_train_list.txt"
        },
        "cityscapes_II_train": {
            "data_dir": "cityscape",
            "data_list": "cityscapes_train_list.txt"
        },
        "cityscapes_self_distill_train": {
            "data_dir": "cityscape",
            "data_list": "cityscapes_train_list.txt",
            "label_dir": "cityscape/syn_ur_soft"
        },
        "cityscapes_val": {
            "data_dir": "cityscape",
            "data_list": "cityscapes_val_list.txt"
        },
        "bdd_val": {
            "data_dir": "BDD/bdd-100k/bdd100k/images/10k/val/",
            "masks_dir": "BDD/bdd-100k/bdd100k/labels/sem_seg/masks/val/"
        },
        "bdd_train": {
            "data_dir": "BDD/bdd-100k/bdd100k/images/10k/train/",
            "masks_dir": "BDD/bdd-100k/bdd100k/labels/sem_seg/masks/train/"
        },
        "bdd_TT_train": {
            "data_dir": "BDD/bdd-100k/bdd100k/images/10k/train/",
            "masks_dir": "BDD/bdd-100k/bdd100k/labels/sem_seg/masks/train/"
        },
        "bdd_II_train": {
            "data_dir": "BDD/bdd-100k/bdd100k/images/10k/train/",
            "masks_dir": "BDD/bdd-100k/bdd100k/labels/sem_seg/masks/train/"
        },
        "SDllama70b": {
            "data_dir": "/data1/zd/dataset/sd_llama70b/images",
            "data_list": "/data1/zd/dataset/sd_llama70b/train_dict.json"
        },
        "suim_train": {
            "data_dir": "SUIM/orin_data/train_val/images/",
            "masks_dir": "SUIM/orin_data/train_val/masks/"
        },
        "suim_TT_train": {
            "data_dir": "SUIM/orin_data/train_val/images/",
            "masks_dir": "SUIM/orin_data/train_val/masks/"
        },
        "suim_II_train": {
            "data_dir": "SUIM/orin_data/train_val/images/",
            "masks_dir": "SUIM/orin_data/train_val/masks/"
        },
        "suim_val": {
            "data_dir": "SUIM/orin_data/TEST/images/",
            "masks_dir": "SUIM/orin_data/TEST/masks/"
        },
        "suim_mixed_train": {
            "data_dir": "SUIM/SUIM_Corrupted/mixed/train_val/images/",
            "masks_dir": "SUIM/SUIM_Corrupted/mixed/train_val/masks/"
        },
        "suim_mixed_val": {
            "data_dir": "SUIM/SUIM_Corrupted/mixed/TEST/images/",
            "masks_dir": "SUIM/SUIM_Corrupted/mixed/TEST/masks/"
        },
        "suim_attenuation_train": {
            "data_dir": "SUIM/SUIM_Corrupted/attenuation/train_val/images/",
            "masks_dir": "SUIM/SUIM_Corrupted/attenuation/train_val/masks/"
        },
        "suim_attenuation_val": {
            "data_dir": "SUIM/SUIM_Corrupted/attenuation/TEST/images/",
            "masks_dir": "SUIM/SUIM_Corrupted/attenuation/TEST/masks/"
        },
        "suim_haze_train": {
            "data_dir": "SUIM/SUIM_Corrupted/haze/train_val/images/",
            "masks_dir": "SUIM/SUIM_Corrupted/haze/train_val/masks/"
        },
        "suim_haze_val": {
            "data_dir": "SUIM/SUIM_Corrupted/haze/TEST/images/",
            "masks_dir": "SUIM/SUIM_Corrupted/haze/TEST/masks/"
        },
        "uiis_train": {
            "data_dir": "UIIS/orin_data/train_val/images/",
            "masks_dir": "UIIS/orin_data/train_val/masks/",
            "label_palette": "uiis",
            "mask_suffixes": (".bmp", ".png")
        },
        "uiis_TT_train": {
            "data_dir": "UIIS/orin_data/train_val/images/",
            "masks_dir": "UIIS/orin_data/train_val/masks/",
            "label_palette": "uiis",
            "mask_suffixes": (".bmp", ".png")
        },
        "uiis_II_train": {
            "data_dir": "UIIS/orin_data/train_val/images/",
            "masks_dir": "UIIS/orin_data/train_val/masks/",
            "label_palette": "uiis",
            "mask_suffixes": (".bmp", ".png")
        },
        "uiis_val": {
            "data_dir": "UIIS/orin_data/TEST/images/",
            "masks_dir": "UIIS/orin_data/TEST/masks/",
            "label_palette": "uiis",
            "mask_suffixes": (".bmp", ".png")
        },
        "uiis10k_train": {
            "data_dir": "UIIS10K/orin_data/train_val/images/",
            "masks_dir": "UIIS10K/orin_data/train_val/masks/",
            "label_palette": "uiis",
            "mask_suffixes": (".bmp",),
            "disabled_reason": (
                "UIIS10K uses a 10-category taxonomy and is not compatible "
                "with the 8-class SUIM/UIIS protocol used in this repo. "
                "Use uiis_train as the source dataset instead."
            )
        },
        "uiis10k_TT_train": {
            "data_dir": "UIIS10K/orin_data/train_val/images/",
            "masks_dir": "UIIS10K/orin_data/train_val/masks/",
            "label_palette": "uiis",
            "mask_suffixes": (".bmp",),
            "disabled_reason": (
                "UIIS10K uses a 10-category taxonomy and is not compatible "
                "with the 8-class SUIM/UIIS protocol used in this repo. "
                "Use uiis_TT_train instead."
            )
        },
        "uiis10k_II_train": {
            "data_dir": "UIIS10K/orin_data/train_val/images/",
            "masks_dir": "UIIS10K/orin_data/train_val/masks/",
            "label_palette": "uiis",
            "mask_suffixes": (".bmp",),
            "disabled_reason": (
                "UIIS10K uses a 10-category taxonomy and is not compatible "
                "with the 8-class SUIM/UIIS protocol used in this repo. "
                "Use uiis_II_train instead."
            )
        },
        "uiis10k_val": {
            "data_dir": "UIIS10K/orin_data/TEST/images/",
            "masks_dir": "UIIS10K/orin_data/TEST/masks/",
            "label_palette": "uiis",
            "mask_suffixes": (".bmp",),
            "disabled_reason": (
                "UIIS10K uses a 10-category taxonomy and is not compatible "
                "with the 8-class SUIM/UIIS protocol used in this repo. "
                "Use uiis_val instead."
            )
        },
        "usis10k_bg_train": {
            "data_dir": "USIS10K/USIS10K/train/",
            "masks_dir": "USIS10K/semantic_bg/train/masks/",
            "label_palette": "suim",
            "mask_suffixes": (".png",)
        },
        "usis10k_bg_val": {
            "data_dir": "USIS10K/USIS10K/val/",
            "masks_dir": "USIS10K/semantic_bg/val/masks/",
            "label_palette": "suim",
            "mask_suffixes": (".png",)
        },
        "usis10k_bg_test": {
            "data_dir": "USIS10K/USIS10K/test/",
            "masks_dir": "USIS10K/semantic_bg/test/masks/",
            "label_palette": "suim",
            "mask_suffixes": (".png",)
        },
        "usis10k_ignore_train": {
            "data_dir": "USIS10K/USIS10K/train/",
            "masks_dir": "USIS10K/semantic_ignore/train/masks/",
            "label_palette": "suim",
            "mask_suffixes": (".png",)
        },
        "usis10k_ignore_val": {
            "data_dir": "USIS10K/USIS10K/val/",
            "masks_dir": "USIS10K/semantic_ignore/val/masks/",
            "label_palette": "suim",
            "mask_suffixes": (".png",)
        },
        "usis10k_ignore_test": {
            "data_dir": "USIS10K/USIS10K/test/",
            "masks_dir": "USIS10K/semantic_ignore/test/masks/",
            "label_palette": "suim",
            "mask_suffixes": (".png",)
        },
        "coralscapes_train": {
            "data_dir": "coralscapes/orin_data/train/images/",
            "masks_dir": "coralscapes/orin_data/train/labels/"
        },
        "coralscapes_TT_train": {
            "data_dir": "coralscapes/orin_data/train/images/",
            "masks_dir": "coralscapes/orin_data/train/labels/"
        },
        "coralscapes_II_train": {
            "data_dir": "coralscapes/orin_data/train/images/",
            "masks_dir": "coralscapes/orin_data/train/labels/"
        },
        "coralscapes_val": {
            "data_dir": "coralscapes/orin_data/validation/images/",
            "masks_dir": "coralscapes/orin_data/validation/labels/"
        },
        "coralscapes_test": {
            "data_dir": "coralscapes/orin_data/test/images/",
            "masks_dir": "coralscapes/orin_data/test/labels/"
        },
        "coralscapes_mixed_train": {
            "data_dir": "coralscapes/CoralScapes_Corrupted/mixed/train/images/",
            "masks_dir": "coralscapes/CoralScapes_Corrupted/mixed/train/labels/"
        },
        "coralscapes_mixed_TT_train": {
            "data_dir": "coralscapes/CoralScapes_Corrupted/mixed/train/images/",
            "masks_dir": "coralscapes/CoralScapes_Corrupted/mixed/train/labels/"
        },
        "coralscapes_mixed_II_train": {
            "data_dir": "coralscapes/CoralScapes_Corrupted/mixed/train/images/",
            "masks_dir": "coralscapes/CoralScapes_Corrupted/mixed/train/labels/"
        },
        "coralscapes_mixed_val": {
            "data_dir": "coralscapes/CoralScapes_Corrupted/mixed/validation/images/",
            "masks_dir": "coralscapes/CoralScapes_Corrupted/mixed/validation/labels/"
        },
        "coralscapes_mixed_test": {
            "data_dir": "coralscapes/CoralScapes_Corrupted/mixed/test/images/",
            "masks_dir": "coralscapes/CoralScapes_Corrupted/mixed/test/labels/"
        },
        "coralscapes_attenuation_train": {
            "data_dir": "coralscapes/CoralScapes_Corrupted/attenuation/train/images/",
            "masks_dir": "coralscapes/CoralScapes_Corrupted/attenuation/train/labels/"
        },
        "coralscapes_attenuation_TT_train": {
            "data_dir": "coralscapes/CoralScapes_Corrupted/attenuation/train/images/",
            "masks_dir": "coralscapes/CoralScapes_Corrupted/attenuation/train/labels/"
        },
        "coralscapes_attenuation_II_train": {
            "data_dir": "coralscapes/CoralScapes_Corrupted/attenuation/train/images/",
            "masks_dir": "coralscapes/CoralScapes_Corrupted/attenuation/train/labels/"
        },
        "coralscapes_attenuation_val": {
            "data_dir": "coralscapes/CoralScapes_Corrupted/attenuation/validation/images/",
            "masks_dir": "coralscapes/CoralScapes_Corrupted/attenuation/validation/labels/"
        },
        "coralscapes_attenuation_test": {
            "data_dir": "coralscapes/CoralScapes_Corrupted/attenuation/test/images/",
            "masks_dir": "coralscapes/CoralScapes_Corrupted/attenuation/test/labels/"
        },
        "coralscapes_haze_train": {
            "data_dir": "coralscapes/CoralScapes_Corrupted/haze/train/images/",
            "masks_dir": "coralscapes/CoralScapes_Corrupted/haze/train/labels/"
        },
        "coralscapes_haze_TT_train": {
            "data_dir": "coralscapes/CoralScapes_Corrupted/haze/train/images/",
            "masks_dir": "coralscapes/CoralScapes_Corrupted/haze/train/labels/"
        },
        "coralscapes_haze_II_train": {
            "data_dir": "coralscapes/CoralScapes_Corrupted/haze/train/images/",
            "masks_dir": "coralscapes/CoralScapes_Corrupted/haze/train/labels/"
        },
        "coralscapes_haze_val": {
            "data_dir": "coralscapes/CoralScapes_Corrupted/haze/validation/images/",
            "masks_dir": "coralscapes/CoralScapes_Corrupted/haze/validation/labels/"
        },
        "coralscapes_haze_test": {
            "data_dir": "coralscapes/CoralScapes_Corrupted/haze/test/images/",
            "masks_dir": "coralscapes/CoralScapes_Corrupted/haze/test/labels/"
        },
    }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None, rsc=None, cfg=None):
        if "SDllama70b" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                    root=attrs["data_dir"],
                    data_list=attrs["data_list"])
            return SDllama70bDataSet(args["root"], args["data_list"], max_iters=None, num_classes=num_classes, split=mode, transform=transform)

        if "gta5" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_dir"], attrs["data_list"]),
            )
            return GTA5DataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        elif "synthia" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return synthiaDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if 'distill' in name:
                args['label_dir'] = os.path.join(data_dir, attrs["label_dir"])
                return cityscapesSelfDistillDataSet(args["root"], args["data_list"], args['label_dir'], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
            if 'TT' in name: # only for generate psd
                return cityscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, pseudo=True, cfg=cfg)
            if 'II' in name: 
                return cityscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, pseudo=False)
            if 'meta_val' in mode:
                return cityscapesDataSetMeta(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split='train', transform=transform, rsc=rsc, pseudo=True, cfg=cfg)
            if 'train' in mode:
                return cityscapesDataSetTrain(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, rsc=rsc, cfg=cfg)
            else:
                return cityscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)

        elif "bdd" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                mask_path=os.path.join(data_dir, attrs["masks_dir"]),
            )
            if 'distill' in name:
                RuntimeError("Dataset not available: {}".format(name))
                args['label_dir'] = os.path.join(data_dir, attrs["label_dir"])
                #return cityscapesSelfDistillDataSet(args["root"], args["data_list"], args['label_dir'], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
            if 'TT' in name: # only for generate psd
                return BddDataSet(args["root"], args["mask_path"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, pseudo=True, cfg=cfg)
            if 'II' in name: 
                return BddDataSet(args["root"], args["mask_path"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, pseudo=False)
            if 'meta_val' in mode:
                return BddDataSetMeta(args["root"], args["mask_path"], max_iters=max_iters, num_classes=num_classes, split='train', transform=transform, rsc=rsc, pseudo=True, cfg=cfg)

            if 'train' in mode:
                return BddDataSetTrain(args["root"], args["mask_path"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, rsc=rsc, cfg=cfg)
            else:
                return BddDataSet(args["root"], args["mask_path"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)

        elif any(token in name for token in ("suim", "uiis", "usis10k")):
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            if attrs.get("disabled_reason"):
                raise RuntimeError(
                    "Dataset '{}' is disabled: {}".format(name, attrs["disabled_reason"])
                )
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                mask_path=os.path.join(data_dir, attrs["masks_dir"]),
                label_palette=attrs.get("label_palette", "suim"),
                mask_suffixes=attrs.get("mask_suffixes", (".bmp", ".png")),
            )
            if 'TT' in name:
                return SuimDataSet(args["root"], args["mask_path"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, pseudo=True, cfg=cfg, label_palette=args["label_palette"], mask_suffixes=args["mask_suffixes"])
            if 'II' in name:
                return SuimDataSet(args["root"], args["mask_path"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, pseudo=False, cfg=cfg, label_palette=args["label_palette"], mask_suffixes=args["mask_suffixes"])
            if 'train' in mode:
                return SuimDataSetTrain(args["root"], args["mask_path"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, rsc=rsc, cfg=cfg, label_palette=args["label_palette"], mask_suffixes=args["mask_suffixes"])
            
            return SuimDataSet(args["root"], args["mask_path"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, cfg=cfg, label_palette=args["label_palette"], mask_suffixes=args["mask_suffixes"])

        elif "coralscapes" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                mask_path=os.path.join(data_dir, attrs["masks_dir"]),
            )
            if 'TT' in name:
                return CoralScapesDataSet(args["root"], args["mask_path"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, pseudo=True, cfg=cfg)
            if 'II' in name:
                return CoralScapesDataSet(args["root"], args["mask_path"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, pseudo=False, cfg=cfg)
            if 'train' in mode and cfg is not None:
                return CoralScapesDataSetTrain(args["root"], args["mask_path"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, rsc=rsc, cfg=cfg)

            return CoralScapesDataSet(args["root"], args["mask_path"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, cfg=cfg)

 
        raise RuntimeError("Dataset not available: {}".format(name))


def _register_coralscapes_corrupted_variants():
    variants = {
        "d0055": "CoralScapes_Corrupted",
        "d0072": "CoralScapes_Corrupted_d0072",
        "d0130": "CoralScapes_Corrupted_d0130",
        "hard": "CoralScapes_Corrupted_hard",
    }
    domains = ("mixed", "attenuation", "haze")
    for variant, root in variants.items():
        for domain in domains:
            name = "coralscapes_{}_{}".format(variant, domain)
            base_dir = "coralscapes/{}/{}/".format(root, domain)
            DatasetCatalog.DATASETS.update(
                {
                    "{}_train".format(name): {
                        "data_dir": base_dir + "train/images/",
                        "masks_dir": base_dir + "train/labels/",
                    },
                    "{}_TT_train".format(name): {
                        "data_dir": base_dir + "train/images/",
                        "masks_dir": base_dir + "train/labels/",
                    },
                    "{}_II_train".format(name): {
                        "data_dir": base_dir + "train/images/",
                        "masks_dir": base_dir + "train/labels/",
                    },
                    "{}_val".format(name): {
                        "data_dir": base_dir + "validation/images/",
                        "masks_dir": base_dir + "validation/labels/",
                    },
                    "{}_test".format(name): {
                        "data_dir": base_dir + "test/images/",
                        "masks_dir": base_dir + "test/labels/",
                    },
                }
            )


_register_coralscapes_corrupted_variants()

