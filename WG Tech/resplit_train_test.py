"""
Pools all images from datasets/stud_roboflow_v1 (both train/ and valid/,
already remapped to the 6-class scheme) and re-splits into train/ and
test/ folders, with per-class stratification so thin classes don't
randomly end up missing from one split.

Stratification rule:
  - Classes with >= 5 instances: ~80/20 train/test split
  - Classes with < 5 instances: ALL instances go to train (not enough to
    split meaningfully)
  - Classes with 0 instances (e.g. White_Patches in this source): reported
    as 0, nothing to assign

Usage:
    python resplit_train_test.py
"""

import os
import random
import shutil
from collections import defaultdict

SRC = "datasets/stud_roboflow_v1"
DST = "stud_yolo_dataset_v3"
TEST_FRACTION = 0.2
MIN_INSTANCES_TO_SPLIT = 5
SEED = 42

CLASS_NAMES = ["Black_Patches", "Thread_Missing", "White_Paint_On_Thread", "White_Patches", "White_Patch_Missing", "Okay_Part"]

# stud_roboflow_v1 uses "train" and "valid" as its existing split folder names
SOURCE_SPLIT_FOLDERS = {"train": "train", "valid": "valid"}


def gather_pool():
    """Combine train/ + valid/ (images + labels) into one pool."""
    pool = {}  # base_name -> (img_path, lbl_path or None, fname)
    for split_folder in SOURCE_SPLIT_FOLDERS.values():
        img_dir = os.path.join(SRC, split_folder, "images")
        lbl_dir = os.path.join(SRC, split_folder, "labels")
        if not os.path.isdir(img_dir):
            print(f"WARNING: {img_dir} not found, skipping")
            continue
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            base = os.path.splitext(fname)[0]
            lbl_path = os.path.join(lbl_dir, base + ".txt")
            pool[base] = (
                os.path.join(img_dir, fname),
                lbl_path if os.path.exists(lbl_path) else None,
                fname,
            )
    return pool


def get_classes_in_image(lbl_path):
    if lbl_path is None:
        return set()
    classes = set()
    with open(lbl_path) as f:
        for line in f:
            parts = line.split()
            if parts:
                classes.add(int(parts[0]))
    return classes


def main():
    random.seed(SEED)
    pool = gather_pool()
    print(f"Pooled {len(pool)} unique images from {SRC} (train + valid)\n")

    class_to_images = defaultdict(list)
    for base, (img_path, lbl_path, fname) in pool.items():
        for c in get_classes_in_image(lbl_path):
            class_to_images[c].append(base)

    assigned = {}

    for c in range(len(CLASS_NAMES)):
        imgs = list(set(class_to_images.get(c, [])))
        random.shuffle(imgs)
        unassigned = [b for b in imgs if b not in assigned]

        if len(imgs) == 0:
            print(f"{CLASS_NAMES[c]}: 0 images in this source - nothing to assign")
        elif len(imgs) < MIN_INSTANCES_TO_SPLIT:
            for b in unassigned:
                assigned[b] = "train"
            print(f"{CLASS_NAMES[c]}: {len(imgs)} images (< {MIN_INSTANCES_TO_SPLIT}) -> all to train")
        else:
            n_test = max(1, int(len(imgs) * TEST_FRACTION))
            test_set = set(imgs[:n_test])
            for b in unassigned:
                assigned[b] = "test" if b in test_set else "train"
            print(f"{CLASS_NAMES[c]}: {len(imgs)} images -> ~{n_test} to test, rest to train")

    for base in pool:
        if base not in assigned:
            assigned[base] = "train"

    for split in ["train", "test"]:
        os.makedirs(os.path.join(DST, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DST, "labels", split), exist_ok=True)

    for base, split in assigned.items():
        img_path, lbl_path, fname = pool[base]
        shutil.copy(img_path, os.path.join(DST, "images", split, fname))
        if lbl_path:
            shutil.copy(lbl_path, os.path.join(DST, "labels", split, base + ".txt"))

    print("\nFinal class counts per split:")
    for split in ["train", "test"]:
        counts = {i: 0 for i in range(len(CLASS_NAMES))}
        lbl_dir = os.path.join(DST, "labels", split)
        for f in os.listdir(lbl_dir):
            for line in open(os.path.join(lbl_dir, f)):
                parts = line.split()
                if parts:
                    counts[int(parts[0])] += 1
        print(f"  {split}:")
        for i, n in enumerate(CLASS_NAMES):
            print(f"    {n}: {counts[i]}")

    print(f"\nDone. Output at {DST}/")


if __name__ == "__main__":
    main()