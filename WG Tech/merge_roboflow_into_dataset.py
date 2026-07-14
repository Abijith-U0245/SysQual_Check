"""
Merges datasets/stud_roboflow_v1 (train+valid) into stud_yolo_dataset_v2
(train+val), which was built from dataset/Stud via organize_yolo_dataset.py.

Both sources are assumed to already use the same 6-class index scheme:
  0 Black_Patches, 1 Thread_Missing, 2 White_Paint_On_Thread,
  3 White_Patches, 4 White_Patch_Missing, 5 Okay_Part

Run from the repo root (WG Tech folder):
    python merge_roboflow_into_dataset.py
"""

import os
import shutil

ROBOFLOW_TRAIN_IMG = "datasets/stud_roboflow_v1/train/images"
ROBOFLOW_TRAIN_LBL = "datasets/stud_roboflow_v1/train/labels"
ROBOFLOW_VALID_IMG = "datasets/stud_roboflow_v1/valid/images"
ROBOFLOW_VALID_LBL = "datasets/stud_roboflow_v1/valid/labels"

DST = "stud_yolo_dataset_v2"


def copy_split(src_img_dir, src_lbl_dir, dst_split):
    dst_img_dir = os.path.join(DST, "images", dst_split)
    dst_lbl_dir = os.path.join(DST, "labels", dst_split)
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    n_copied, n_renamed = 0, 0
    for img_name in os.listdir(src_img_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base, ext = os.path.splitext(img_name)
        dst_img_name = img_name

        # Avoid filename collisions with existing files from dataset/Stud
        if os.path.exists(os.path.join(dst_img_dir, dst_img_name)):
            dst_img_name = f"roboflow_{img_name}"
            n_renamed += 1

        shutil.copy(
            os.path.join(src_img_dir, img_name),
            os.path.join(dst_img_dir, dst_img_name),
        )

        lbl_name = base + ".txt"
        src_lbl_path = os.path.join(src_lbl_dir, lbl_name)
        if os.path.exists(src_lbl_path):
            dst_lbl_name = os.path.splitext(dst_img_name)[0] + ".txt"
            shutil.copy(src_lbl_path, os.path.join(dst_lbl_dir, dst_lbl_name))

        n_copied += 1

    print(f"  {dst_split}: copied {n_copied} images ({n_renamed} renamed to avoid collisions)")


def print_class_counts():
    names = ["Black_Patches", "Thread_Missing", "White_Paint_On_Thread", "White_Patches", "White_Patch_Missing", "Okay_Part"]
    for split in ["train", "val"]:
        counts = {i: 0 for i in range(6)}
        lbl_dir = os.path.join(DST, "labels", split)
        for f in os.listdir(lbl_dir):
            if not f.endswith(".txt"):
                continue
            for line in open(os.path.join(lbl_dir, f)):
                parts = line.split()
                if parts:
                    counts[int(parts[0])] += 1
        print(f"  {split}:")
        for i, n in enumerate(names):
            print(f"    {n}: {counts[i]}")


if __name__ == "__main__":
    if not os.path.isdir(DST):
        raise SystemExit(f"{DST} not found - run organize_yolo_dataset.py first (Step 3).")

    print("Merging roboflow train -> stud_yolo_dataset_v2 train...")
    copy_split(ROBOFLOW_TRAIN_IMG, ROBOFLOW_TRAIN_LBL, "train")

    print("Merging roboflow valid -> stud_yolo_dataset_v2 val...")
    copy_split(ROBOFLOW_VALID_IMG, ROBOFLOW_VALID_LBL, "val")

    print("\nFinal combined class counts:")
    print_class_counts()
    print("\nDone. Now update stud_data.yaml (see Step 5) and retrain.")