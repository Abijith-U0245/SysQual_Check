from pathlib import Path
from collections import defaultdict
import random
import shutil
import math

# ======================================================
# CONFIG
# ======================================================
SOURCE = Path("datasets/stud")
DEST = Path("stud_v5")

TRAIN_RATIO = 0.80
SEED = 42

random.seed(SEED)

# Remove old stud_v5 if it exists
if DEST.exists():
    shutil.rmtree(DEST)

# Create folders
for folder in [
    DEST / "images" / "train",
    DEST / "images" / "test",
    DEST / "labels" / "train",
    DEST / "labels" / "test",
]:
    folder.mkdir(parents=True, exist_ok=True)

image_dir = SOURCE / "train" / "images"
label_dir = SOURCE / "train" / "labels"

# ------------------------------------------------------
# Group images by class
# ------------------------------------------------------
class_images = defaultdict(list)

for img in image_dir.iterdir():

    if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    label = label_dir / (img.stem + ".txt")

    if not label.exists():
        print("Missing label:", img.name)
        continue

    with open(label) as f:
        line = f.readline().strip()

    if not line:
        continue

    cls = int(line.split()[0])

    class_images[cls].append((img, label))

print("\n==============================")
print("Class-wise Split")
print("==============================")

train_total = 0
test_total = 0

for cls in sorted(class_images):

    samples = class_images[cls]

    random.shuffle(samples)

    total = len(samples)

    train_count = math.floor(total * TRAIN_RATIO)
    test_count = total - train_count

    # Guarantee at least one test image if possible
    if total >= 2 and test_count == 0:
        train_count -= 1
        test_count = 1

    train = samples[:train_count]
    test = samples[train_count:]

    print(
        f"Class {cls} -> Total:{total:3d}  Train:{len(train):3d}  Test:{len(test):3d}"
    )

    train_total += len(train)
    test_total += len(test)

    for img, lbl in train:
        shutil.copy2(img, DEST / "images" / "train" / img.name)
        shutil.copy2(lbl, DEST / "labels" / "train" / lbl.name)

    for img, lbl in test:
        shutil.copy2(img, DEST / "images" / "test" / img.name)
        shutil.copy2(lbl, DEST / "labels" / "test" / lbl.name)

print("\n==============================")
print("TOTAL")
print("==============================")
print("Train:", train_total)
print("Test :", test_total)

# ------------------------------------------------------
# Create data.yaml
# ------------------------------------------------------
yaml = """train: images/train
val: images/test

nc: 6

names:
  0: Black_Patches
  1: Thread_Missing
  2: White_Paint_On_Thread
  3: White_Patches
  4: White_Patch_Missing
  5: Okay_Part
"""

with open(DEST / "data.yaml", "w") as f:
    f.write(yaml)

print("\nDone! stud_v5 created successfully.")