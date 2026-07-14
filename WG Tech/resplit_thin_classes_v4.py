"""
Re-splits Thread_Missing (class 1) and White_Patch_Missing (class 4)
properly, now that you've added more real images directly into
stud_yolo_dataset_v4/images/train.

Previously these two classes were duplicated into BOTH train and test
(memorization check only, since there wasn't enough data for a real
split). This script:

1. Removes the old duplicated copies from test/ for these two classes
   (so we don't keep stale leaked test data)
2. Gathers every image in train/ that contains class 1 or class 4
   (this includes your newly added images plus the original ones)
3. If a class now has enough images (>= MIN_INSTANCES_TO_SPLIT), does a
   real ~80/20 split, moving the test portion out of train/ into test/
4. If still below the threshold, leaves it as-is in train only (no
   duplication this time - a proper "not enough data yet" state, more
   honest than a fake memorization number)

Does NOT touch Black_Patches, White_Paint_On_Thread, White_Patches,
Okay_Part - those already have a correct, working split.

Usage:
    python resplit_thin_classes_v4.py
"""

import os
import random
import shutil

DST = "stud_yolo_dataset_v4"
TARGET_CLASSES = {1: "Thread_Missing", 4: "White_Patch_Missing"}
MIN_INSTANCES_TO_SPLIT = 5
TEST_FRACTION = 0.2
SEED = 42


def get_classes_in_label(lbl_path):
    if not os.path.exists(lbl_path):
        return set()
    classes = set()
    with open(lbl_path) as f:
        for line in f:
            parts = line.split()
            if parts:
                classes.add(int(parts[0]))
    return classes


def remove_old_duplicates_from_test():
    """Remove any existing test/ images that are ONLY there because of the old duplicate-class trick."""
    img_dir = os.path.join(DST, "images", "test")
    lbl_dir = os.path.join(DST, "labels", "test")
    removed = 0
    for fname in list(os.listdir(img_dir)):
        base = os.path.splitext(fname)[0]
        lbl_path = os.path.join(lbl_dir, base + ".txt")
        classes = get_classes_in_label(lbl_path)
        # Only remove if the image's classes are a subset of target classes
        # (i.e. it's purely a Thread_Missing/White_Patch_Missing image, not
        # mixed with something else we should keep in test)
        if classes and classes.issubset(TARGET_CLASSES.keys()):
            os.remove(os.path.join(img_dir, fname))
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
            removed += 1
    print(f"Removed {removed} old duplicated test images for {list(TARGET_CLASSES.values())}")


def main():
    random.seed(SEED)
    remove_old_duplicates_from_test()

    train_img_dir = os.path.join(DST, "images", "train")
    train_lbl_dir = os.path.join(DST, "labels", "train")
    test_img_dir = os.path.join(DST, "images", "test")
    test_lbl_dir = os.path.join(DST, "labels", "test")

    for c, name in TARGET_CLASSES.items():
        matching = []
        for fname in os.listdir(train_img_dir):
            base = os.path.splitext(fname)[0]
            lbl_path = os.path.join(train_lbl_dir, base + ".txt")
            if c in get_classes_in_label(lbl_path):
                matching.append((fname, base))

        print(f"\n{name}: {len(matching)} images currently in train")

        if len(matching) < MIN_INSTANCES_TO_SPLIT:
            print(f"  Still below {MIN_INSTANCES_TO_SPLIT} - leaving all in train, no test split yet")
            continue

        random.shuffle(matching)
        n_test = max(1, int(len(matching) * TEST_FRACTION))
        to_move = matching[:n_test]

        for fname, base in to_move:
            shutil.move(os.path.join(train_img_dir, fname), os.path.join(test_img_dir, fname))
            lbl_src = os.path.join(train_lbl_dir, base + ".txt")
            if os.path.exists(lbl_src):
                shutil.move(lbl_src, os.path.join(test_lbl_dir, base + ".txt"))

        print(f"  Moved {len(to_move)} images to test, {len(matching) - len(to_move)} remain in train")

    print("\nFinal class counts per split:")
    names = ["Black_Patches", "Thread_Missing", "White_Paint_On_Thread", "White_Patches", "White_Patch_Missing", "Okay_Part"]
    for split in ["train", "test"]:
        counts = {i: 0 for i in range(6)}
        lbl_dir = os.path.join(DST, "labels", split)
        for f in os.listdir(lbl_dir):
            for line in open(os.path.join(lbl_dir, f)):
                parts = line.split()
                if parts:
                    counts[int(parts[0])] += 1
        print(f"  {split}:")
        for i, n in enumerate(names):
            print(f"    {n}: {counts[i]}")


if __name__ == "__main__":
    main()