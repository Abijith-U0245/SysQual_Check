from pathlib import Path
import shutil

# ===========================
# SOURCE AND DESTINATION
# ===========================

SOURCE = Path("datasets/stud/train")
DEST = Path("stud_full")      # <-- Change this to any folder name you want

# Create folders
for folder in [
    DEST / "images" / "train",
    DEST / "images" / "test",
    DEST / "labels" / "train",
    DEST / "labels" / "test",
]:
    folder.mkdir(parents=True, exist_ok=True)

image_count = 0
label_count = 0
missing_labels = 0

# Copy all images and labels to BOTH train and test
for img in (SOURCE / "images").iterdir():

    if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    label = SOURCE / "labels" / (img.stem + ".txt")

    if not label.exists():
        print("Missing label:", img.name)
        missing_labels += 1
        continue

    # Train
    shutil.copy2(img, DEST / "images" / "train" / img.name)
    shutil.copy2(label, DEST / "labels" / "train" / label.name)

    # Test
    shutil.copy2(img, DEST / "images" / "test" / img.name)
    shutil.copy2(label, DEST / "labels" / "test" / label.name)

    image_count += 1
    label_count += 1

# Create YAML
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

with open(DEST / "stud_data.yaml", "w") as f:
    f.write(yaml)

print("\n===================================")
print("Dataset Created Successfully")
print("===================================")
print(f"Dataset Folder : {DEST}")
print(f"Images Copied  : {image_count}")
print(f"Labels Copied  : {label_count}")
print(f"Missing Labels : {missing_labels}")
print("All images are present in BOTH train and test.")