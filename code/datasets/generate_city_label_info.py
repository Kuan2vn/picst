import os
import pickle
import numpy as np
from PIL import Image


def gen_lb_info(cfg, name):
    """
    Scan pseudo-label PNG files in OUTPUT_DIR/<name>/ and build two mappings:
      label_to_file[class_id]  -> list of filenames containing that class
      file_to_label[filename]  -> dominant class id for that file

    These mappings are used by cityscapesDataSetTrain for class-balanced sampling
    and copy-paste Mixup augmentation.

    The PNG files are palette images saved by get_color_pallete() where
    pixel value == train class id (0..NUM_CLASSES-1) or 255 (ignore).
    """
    folder = os.path.join(cfg.OUTPUT_DIR, name)
    output_path = os.path.join(cfg.OUTPUT_DIR, f"{name}.p")
    num_classes = cfg.MODEL.NUM_CLASSES

    label_to_file = {i: [] for i in range(num_classes)}
    file_to_label = {}

    if not os.path.isdir(folder):
        print(f"[gen_lb_info] WARNING: folder {folder} does not exist, creating empty mapping.")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump((label_to_file, file_to_label), f)
        return

    png_files = [f for f in os.listdir(folder) if f.endswith('.png')]
    if len(png_files) == 0:
        print(f"[gen_lb_info] WARNING: no PNG files found in {folder}.")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump((label_to_file, file_to_label), f)
        return

    for fname in png_files:
        fpath = os.path.join(folder, fname)
        try:
            lbl = np.array(Image.open(fpath))
        except Exception as e:
            print(f"[gen_lb_info] Could not read {fpath}: {e}")
            continue

        # Count pixels per class (ignore 255)
        unique, counts = np.unique(lbl, return_counts=True)
        class_counts = {}
        for cls, cnt in zip(unique, counts):
            if 0 <= cls < num_classes:
                class_counts[int(cls)] = int(cnt)

        if len(class_counts) == 0:
            continue

        # Register this file for each class it contains
        for cls in class_counts:
            label_to_file[cls].append(fname)

        # Store a list of classes present in the file
        file_to_label[fname] = list(class_counts.keys())

    total_files = len(file_to_label)
    total_entries = sum(len(v) for v in label_to_file.values())
    print(f"[gen_lb_info] {name}: {total_files} files scanned, {total_entries} class-file entries built.")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump((label_to_file, file_to_label), f)
    print(f"[gen_lb_info] Saved {output_path}")
