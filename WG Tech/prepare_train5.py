"""
Run this ONCE from inside your repo root, AFTER placing the roboflow export at:
    datasets/stud_roboflow_v1/train/images
    datasets/stud_roboflow_v1/train/labels

What it does:
  1. Fixes class indices in every label .txt (Roboflow's 5-class order -> your 6-class order)
  2. Moves ~15% of images+labels into a new valid/ folder (since Roboflow gave you no valid split)

Usage:
    python prepare_train5.py
"""

import os
import random
import shutil

# ---- CONFIG (edit only if your folder names differ) ----
BASE = "datasets/stud_roboflow_v1"
TRAIN_IMG = os.path.join(BASE, "train/images")
TRAIN_LBL = os.path.join(BASE, "train/labels")
VALID_IMG = os.path.join(BASE, "valid/images")
VALID_LBL = os.path.join(BASE, "valid/labels")

# Roboflow index -> your correct 6-class index
# 0=Black_Patches, 1=Thread_Missing, 2=White_Paint_On_Thread, 3=White_Patches, 4=White_Patch_Missing, 5=Okay_Part
# CONFIRM this matches your real stud_data.yaml class order before running.
CLASS_MAPPING = {
    0: 0,  # Stud_Black_Patches       -> Black_Patches
    1: 5,  # Stud_OKAY                -> Okay_Part
    2: 1,  # Stud_Thread_Missing      -> Thread_Missing
    3: 4,  # Stud_WHITE_PATCH_MISSING -> White_Patch_Missing
    4: 2,  # Stud_pAINT_ON_Thread     -> White_Paint_On_Thread
}

VAL_FRACTION = 0.15
SEED = 42
# ---------------------------------------------------------


def remap_labels():
    print("Step 1: Remapping class indices in label files...")
    count = 0
    for fname in os.listdir(TRAIN_LBL):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(TRAIN_LBL, fname)
        with open(path) as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            old_cls = int(parts[0])
            if old_cls not in CLASS_MAPPING:
                print(f"  WARNING: unexpected class {old_cls} in {fname}, left unchanged")
                new_lines.append(" ".join(parts))
                continue
            parts[0] = str(CLASS_MAPPING[old_cls])
            new_lines.append(" ".join(parts))

        with open(path, "w") as f:
            f.write("\n".join(new_lines) + "\n")
        count += 1

    print(f"  Remapped {count} label files.\n")


def make_valid_split():
    print("Step 2: Creating valid/ split...")
    os.makedirs(VALID_IMG, exist_ok=True)
    os.makedirs(VALID_LBL, exist_ok=True)

    random.seed(SEED)
    images = [f for f in os.listdir(TRAIN_IMG) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)
    val_count = max(1, int(len(images) * VAL_FRACTION))
    val_set = images[:val_count]

    moved = 0
    for img_name in val_set:
        base = os.path.splitext(img_name)[0]
        shutil.move(os.path.join(TRAIN_IMG, img_name), os.path.join(VALID_IMG, img_name))

        lbl_name = base + ".txt"
        lbl_path = os.path.join(TRAIN_LBL, lbl_name)
        if os.path.exists(lbl_path):
            shutil.move(lbl_path, os.path.join(VALID_LBL, lbl_name))
        moved += 1

    print(f"  Moved {moved} images (+ matching labels) to valid/, out of {len(images)} total.\n")


def print_class_counts():
    print("Step 3: Class counts per split (sanity check)...")
    names = ['Black_Patches', 'Thread_Missing', 'White_Paint_On_Thread', 'White_Patches', 'White_Patch_Missing', 'Okay_Part']
    for split, lbl_dir in [("train", TRAIN_LBL), ("valid", VALID_LBL)]:
        counts = {i: 0 for i in range(6)}
        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(lbl_dir, fname)) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        counts[int(parts[0])] += 1
        print(f"  {split}:")
        for i, n in enumerate(names):
            print(f"    {n}: {counts[i]}")
    print()


if __name__ == "__main__":
    if not os.path.isdir(TRAIN_IMG) or not os.path.isdir(TRAIN_LBL):
        raise SystemExit(
            f"Could not find {TRAIN_IMG} or {TRAIN_LBL}. "
            f"Place the Roboflow export at datasets/stud_roboflow_v1/train/ first."
        )

    remap_labels()
    make_valid_split()
    print_class_counts()
    print("Done. Now copy stud_data_roboflow.yaml to your repo root and run train5.")