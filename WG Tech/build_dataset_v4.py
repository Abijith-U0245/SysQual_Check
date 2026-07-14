"""
Builds stud_yolo_dataset_v4 from datasets/stud_roboflow_v1 (train + valid
pooled).

By this point the White_Patches polygon/mislabel issue has already been
fixed by fix_white_patches_polygons.py (run that first, separately).

SPLIT RULE:
   - Black_Patches, White_Paint_On_Thread, White_Patches, Okay_Part
     (classes 0, 2, 3, 5): normal ~80/20 train/test split - all have
     enough real images now.
   - Thread_Missing, White_Patch_Missing (classes 1, 4): put ALL images
     into BOTH train and test. This is a memorization check, not a real
     generalization test - it lets you see if the model learns these
     classes at all, until you have enough real images to do a proper
     held-out split.

Usage:
    python build_dataset_v4.py
"""

import os
import random
import shutil
from collections import defaultdict

SRC = "datasets/stud_roboflow_v1"
DST = "stud_yolo_dataset_v4"
TEST_FRACTION = 0.2
SEED = 42

CLASS_NAMES = ["Black_Patches", "Thread_Missing", "White_Paint_On_Thread", "White_Patches", "White_Patch_Missing", "Okay_Part"]

# Classes that get the "duplicate into both train and test" treatment
# (still too thin for a real held-out split)
DUPLICATE_CLASSES = {1, 4}  # Thread_Missing, White_Patch_Missing

SOURCE_SPLIT_FOLDERS = ["train", "valid"]


def gather_pool():
    pool = {}
    for split_folder in SOURCE_SPLIT_FOLDERS:
        img_dir = os.path.join(SRC, split_folder, "images")
        lbl_dir = os.path.join(SRC, split_folder, "labels")
        if not os.path.isdir(img_dir):
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
    print(f"Pooled {len(pool)} unique images from {SRC}\n")

    class_to_images = defaultdict(list)
    for base, (img_path, lbl_path, fname) in pool.items():
        for c in get_classes_in_image(lbl_path):
            class_to_images[c].append(base)

    train_set, test_set = set(), set()
    normal_assigned = set()

    for c in range(len(CLASS_NAMES)):
        if c in DUPLICATE_CLASSES:
            continue
        imgs = list(set(class_to_images.get(c, [])))
        random.shuffle(imgs)
        unassigned = [b for b in imgs if b not in normal_assigned]
        n_test = max(1, int(len(imgs) * TEST_FRACTION)) if imgs else 0
        test_subset = set(imgs[:n_test])
        for b in unassigned:
            if b in test_subset:
                test_set.add(b)
            else:
                train_set.add(b)
            normal_assigned.add(b)
        print(f"{CLASS_NAMES[c]}: {len(imgs)} images -> {n_test} to test, rest to train (normal split)")

    for c in DUPLICATE_CLASSES:
        imgs = list(set(class_to_images.get(c, [])))
        for b in imgs:
            train_set.add(b)
            test_set.add(b)
        print(f"{CLASS_NAMES[c]}: {len(imgs)} images -> ALL copied to BOTH train and test (memorization check only)")

    for base in pool:
        if base not in train_set and base not in test_set:
            train_set.add(base)

    for split in ["train", "test"]:
        os.makedirs(os.path.join(DST, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DST, "labels", split), exist_ok=True)

    for base in train_set:
        img_path, lbl_path, fname = pool[base]
        shutil.copy(img_path, os.path.join(DST, "images", "train", fname))
        if lbl_path:
            shutil.copy(lbl_path, os.path.join(DST, "labels", "train", base + ".txt"))

    for base in test_set:
        img_path, lbl_path, fname = pool[base]
        shutil.copy(img_path, os.path.join(DST, "images", "test", fname))
        if lbl_path:
            shutil.copy(lbl_path, os.path.join(DST, "labels", "test", base + ".txt"))

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
            tag = "  <-- duplicated (memorization check only)" if i in DUPLICATE_CLASSES else ""
            print(f"    {n}: {counts[i]}{tag}")

    print(f"\nDone. Output at {DST}/")


if __name__ == "__main__":
    main()