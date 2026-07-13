"""
Fixes class-index mismatches caused by every annotation folder having a
different local classes.txt. LabelImg writes a NUMBER into each .txt label
file based on that folder's own classes.txt order — if folders disagree on
order (or only list a subset of classes), the same number silently means
different things once everything is merged for training.

APPROACH: rather than trying to decode each folder's (proven unreliable —
some folders' classes.txt was clearly copy-pasted/reused from a different
folder's session) local classes.txt, this trusts the one thing that IS
reliable: every image was manually sorted into a folder named after its
actual defect type. So every box in a given folder gets forced to that
folder's correct global class index, regardless of what number was
originally written.

This is safe as long as each folder genuinely contains only one real
defect type throughout (true for this dataset — confirmed each folder's
raw indices were monolithic, i.e. one class per folder, not a mix).

Run this once, then re-run organize_yolo_dataset.py and retrain from
scratch — the previous best.pt was trained on the corrupted mapping and
can't be salvaged by fine-tuning, it needs a clean retrain.

Usage:
    python fix_class_indices.py --src "dataset/Stud"
"""

import argparse
import os

# The one correct, canonical mapping every folder should use.
GLOBAL_CLASSES = ["Black_Patches", "Thread_Missing", "White_Paint_On_Thread", "White_Patches"]

# Maps folder name (lowercased) -> its real class. Add entries here for
# other parts/folders as you extend this to Pin/Adaptor/Capnut.
FOLDER_NAME_TO_CLASS = {
    "black patches": "Black_Patches",
    "thread missing": "Thread_Missing",
    "white paint on thread": "White_Paint_On_Thread",
    "white patches": "White_Patches",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Folder containing the per-class subfolders")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change without writing files")
    args = parser.parse_args()

    global_index = {name: i for i, name in enumerate(GLOBAL_CLASSES)}

    for folder_name in sorted(os.listdir(args.src)):
        folder_path = os.path.join(args.src, folder_name)
        if not os.path.isdir(folder_path):
            continue

        canonical_name = FOLDER_NAME_TO_CLASS.get(folder_name.lower())
        print(f"\n=== {folder_name} ===")

        if canonical_name is None:
            # Not a recognized defect folder (e.g. Okay_Part, White Patch
            # Missing — these are background/unlabeled folders, nothing to fix)
            print("  not a mapped defect class — skipping (expected for background folders)")
            continue

        correct_idx = global_index[canonical_name]
        print(f"  forcing every box in this folder to class {correct_idx} ({canonical_name})")

        n_fixed, n_unchanged = 0, 0

        for fname in os.listdir(folder_path):
            if not fname.endswith(".txt") or fname == "classes.txt":
                continue
            fpath = os.path.join(folder_path, fname)
            lines = open(fpath).read().strip().splitlines()
            if not lines:
                continue

            new_lines = []
            changed = False
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    new_lines.append(line)
                    continue
                old_idx = int(parts[0])
                if old_idx != correct_idx:
                    changed = True
                new_lines.append(" ".join([str(correct_idx)] + parts[1:]))

            if changed:
                n_fixed += 1
                if not args.dry_run:
                    with open(fpath, "w") as f:
                        f.write("\n".join(new_lines) + "\n")
            else:
                n_unchanged += 1

        print(f"  files corrected: {n_fixed}   already correct: {n_unchanged}")

        # Overwrite this folder's classes.txt with the correct global one
        # so future LabelImg sessions here stay consistent.
        if not args.dry_run:
            with open(os.path.join(folder_path, "classes.txt"), "w") as f:
                f.write("\n".join(GLOBAL_CLASSES) + "\n")

    print("\nDone. Re-run organize_yolo_dataset.py and retrain from scratch —")
    print("the previous best.pt was trained on the corrupted mapping.")


if __name__ == "__main__":
    main()