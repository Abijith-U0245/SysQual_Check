"""
Fixes the real White_Patches problem: these label files contain POLYGON
(segmentation) coordinates - class followed by many x,y pairs - not
bounding boxes. YOLO's detection trainer can't use these as-is, so they
were being silently dropped, leaving White_Patches with 0 usable
instances in every training run so far.

This script:
1. Finds every label file where a line has more than 5 numbers (polygon)
2. Computes the bounding box (min/max x, min/max y) that contains all
   the polygon points
3. Writes that as a proper 5-number YOLO detection line: class x_center
   y_center width height
4. Also forces the class index to 3 (White_Patches) for any file whose
   filename starts with "White Patches", since these were also using
   class 2 (White_Paint_On_Thread) incorrectly

Run this on the SOURCE folders (datasets/stud_roboflow_v1) so the fix is
permanent, then rerun build_dataset_v4.py to rebuild the dataset with
corrected labels.

Usage:
    python fix_white_patches_polygons.py
"""

import os

SRC = "datasets/stud_roboflow_v1"
SPLIT_FOLDERS = ["train", "valid"]
WHITE_PATCHES_CLASS = 3


def polygon_to_bbox(coords):
    """coords: flat list of x1,y1,x2,y2,... (normalized 0-1). Returns (xc, yc, w, h)."""
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    xc = x_min + w / 2
    yc = y_min + h / 2
    return xc, yc, w, h


def main():
    total_converted = 0
    total_files = 0

    for split_folder in SPLIT_FOLDERS:
        lbl_dir = os.path.join(SRC, split_folder, "labels")
        if not os.path.isdir(lbl_dir):
            continue

        for fname in os.listdir(lbl_dir):
            if not fname.startswith("White Patches") or not fname.endswith(".txt"):
                continue

            fpath = os.path.join(lbl_dir, fname)
            lines = open(fpath).read().strip().splitlines()
            new_lines = []
            file_changed = False

            for line in lines:
                parts = line.split()
                if not parts:
                    continue

                if len(parts) == 5:
                    # Already a bbox line - just force correct class
                    if parts[0] != str(WHITE_PATCHES_CLASS):
                        parts[0] = str(WHITE_PATCHES_CLASS)
                        file_changed = True
                    new_lines.append(" ".join(parts))
                elif len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                    # Polygon line - convert to bbox
                    coords = [float(v) for v in parts[1:]]
                    xc, yc, w, h = polygon_to_bbox(coords)
                    new_lines.append(f"{WHITE_PATCHES_CLASS} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    file_changed = True
                    total_converted += 1
                else:
                    print(f"  WARNING: unexpected format in {fname}, line kept as-is: {line[:50]}...")
                    new_lines.append(line)

            if file_changed:
                with open(fpath, "w") as f:
                    f.write("\n".join(new_lines) + "\n")
                total_files += 1

    print(f"Converted {total_converted} polygon annotations to bounding boxes")
    print(f"Updated {total_files} label files total")
    print("\nDone. Now rerun build_dataset_v4.py to rebuild the dataset with corrected labels.")


if __name__ == "__main__":
    main()