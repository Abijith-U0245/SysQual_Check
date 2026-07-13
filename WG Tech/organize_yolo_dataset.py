"""
Splits your per-class annotated images (images + matching YOLO .txt labels
in the same folder) into the images/train, images/val, labels/train,
labels/val structure Ultralytics expects.

Also drops exact byte-for-byte duplicate images within each class before
splitting — a duplicate that lands in both train and val makes your
validation accuracy meaningless (the model would just be recognizing an
image it already memorized during training).

Usage:
    python organize_yolo_dataset.py --src dataset --dst yolo_dataset --val-split 0.2
    python organize_yolo_dataset.py --src "dataset/Stud" --dst stud_yolo_dataset --val-split 0.2
"""

import argparse
import hashlib
import os
import random
import shutil


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Folder containing per-class subfolders of images+labels")
    parser.add_argument("--dst", required=True, help="Output folder for the YOLO train/val structure")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(args.dst, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.dst, "labels", split), exist_ok=True)

    counts = {}
    for class_folder in sorted(os.listdir(args.src)):
        class_path = os.path.join(args.src, class_folder)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        # Drop exact duplicates (same file content, different filename —
        # e.g. "img.jpg" and "img - Copy.jpg")
        seen_hashes = {}
        unique_images = []
        n_dupes = 0
        for img_name in images:
            h = file_hash(os.path.join(class_path, img_name))
            if h in seen_hashes:
                n_dupes += 1
                continue
            seen_hashes[h] = img_name
            unique_images.append(img_name)

        images = unique_images
        random.shuffle(images)
        n_val = max(1, int(len(images) * args.val_split)) if images else 0

        for i, img_name in enumerate(images):
            split = "val" if i < n_val else "train"
            label_name = os.path.splitext(img_name)[0] + ".txt"

            src_img = os.path.join(class_path, img_name)
            src_label = os.path.join(class_path, label_name)

            shutil.copy(src_img, os.path.join(args.dst, "images", split, img_name))
            if os.path.exists(src_label):
                shutil.copy(src_label, os.path.join(args.dst, "labels", split, label_name))
            else:
                print(f"NOTE: no label file for {src_img} — copied as an unlabeled/background image (fine if intentional)")

        counts[class_folder] = (len(images), n_dupes)

    print("\nUnique images per class folder (duplicates dropped):")
    for k, (v, dupes) in counts.items():
        flag = "  <-- LOW, consider collecting more real images" if v < 50 else ""
        dupe_note = f"  ({dupes} exact duplicates skipped)" if dupes else ""
        print(f"  {k}: {v}{dupe_note}{flag}")
    print(f"\nDone. Output at {args.dst}/")


if __name__ == "__main__":
    main()