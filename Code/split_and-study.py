import re
import json
import shutil
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import train_test_split

# ---------------- CONFIG ----------------
INPUT_DIR = Path("../Project-SOS-Signal/data/padded_data")          # output of zero_padding.py
OUTPUT_DIR = Path("../Project-SOS-Signal/data/split_data")          # will contain train/ and test/
TEST_SIZE = 0.2                          # 20% of GROUPS go to test
RANDOM_SEED = 42
# -----------------------------------------

# Matches: <prefix>__<frame_number>.npy
FRAME_PATTERN = re.compile(r"^(.*)__(\d+)\.npy$")

# Matches an optional "_augN" suffix at the end of the hex/prefix part
AUG_SUFFIX_PATTERN = re.compile(r"^(.*)_aug\d+$")


def find_npy_files(root: Path):
    return list(root.rglob("*.npy"))


def parse_filename(f: Path):
    """
    Returns (label, hex_and_aug, frame_idx) for a file like:
    not_sos__1766234603_f2a6d95d_aug1__0.npy
    -> label='not_sos', hex_and_aug='1766234603_f2a6d95d_aug1', frame_idx=0
    """
    match = FRAME_PATTERN.match(f.name)
    if not match:
        return None
    prefix, frame_idx = match.group(1), int(match.group(2))

    # prefix is "label__hex_and_aug" -- split on the FIRST "__"
    parts = prefix.split("__", 1)
    if len(parts) != 2:
        return None
    label, hex_and_aug = parts
    return label, hex_and_aug, frame_idx


def base_hexcode(hex_and_aug: str) -> str:
    """Strip trailing _augN so original + all its augmentations share one key."""
    m = AUG_SUFFIX_PATTERN.match(hex_and_aug)
    return m.group(1) if m else hex_and_aug


def build_group_index(files):
    """
    Returns:
      sequences: dict[(rel_dir, label, hex_and_aug)] -> list of file paths (its 90 frames)
      group_label: dict[base_hex] -> label  (assumes label is consistent within a base_hex group)
      group_sequences: dict[base_hex] -> list of (rel_dir, label, hex_and_aug) sequence keys
    """
    sequences = defaultdict(list)
    group_label = {}
    group_sequences = defaultdict(list)

    for f in files:
        parsed = parse_filename(f)
        if parsed is None:
            print(f"⚠️  Skipping unrecognized filename: {f.name}")
            continue
        label, hex_and_aug, frame_idx = parsed
        rel_dir = f.parent.relative_to(INPUT_DIR)
        seq_key = (rel_dir, label, hex_and_aug)
        sequences[seq_key].append(f)

        b_hex = base_hexcode(hex_and_aug)
        if b_hex in group_label and group_label[b_hex] != label:
            print(f"Label mismatch within group {b_hex}: "
                  f"{group_label[b_hex]} vs {label} (check filenames)")
        group_label[b_hex] = label
        if seq_key not in group_sequences[b_hex]:
            group_sequences[b_hex].append(seq_key)

    return sequences, group_label, group_sequences


def copy_sequences(seq_keys, sequences, split_name):
    """Copy every frame file for the given sequence keys into OUTPUT_DIR/split_name/..."""
    for seq_key in seq_keys:
        rel_dir, label, hex_and_aug = seq_key
        out_dir = OUTPUT_DIR / split_name / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in sequences[seq_key]:
            shutil.copy2(f, out_dir / f.name)


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR.resolve()}")

    files = find_npy_files(INPUT_DIR)
    print(f"Found {len(files)} .npy files under {INPUT_DIR}")

    sequences, group_label, group_sequences = build_group_index(files)

    base_hexes = sorted(group_sequences.keys())
    labels_for_split = [group_label[h] for h in base_hexes]

    print(f"\nTotal frame files:        {len(files)}")
    print(f"Total sequences (aug incl.): {len(sequences)}")
    print(f"Total groups (orig+augs):    {len(base_hexes)}")

    label_counts = defaultdict(int)
    for lbl in labels_for_split:
        label_counts[lbl] += 1
    print(f"Group label counts: {dict(label_counts)}\n")

    # ---- Group-aware, label-stratified split on BASE HEXCODE ----
    train_hexes, test_hexes = train_test_split(
        base_hexes,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels_for_split,
    )
    train_hexes, test_hexes = set(train_hexes), set(test_hexes)

    # Sanity check: no overlap
    assert train_hexes.isdisjoint(test_hexes), "Group leakage detected between train/test!"

    # Expand groups -> full sequence keys (original + all augmentations)
    train_seq_keys = [sk for h in train_hexes for sk in group_sequences[h]]
    test_seq_keys = [sk for h in test_hexes for sk in group_sequences[h]]

    print(f"Train groups: {len(train_hexes)} -> {len(train_seq_keys)} sequences "
          f"-> {sum(len(sequences[sk]) for sk in train_seq_keys)} frame files")
    print(f"Test groups:  {len(test_hexes)} -> {len(test_seq_keys)} sequences "
          f"-> {sum(len(sequences[sk]) for sk in test_seq_keys)} frame files")

    # ---- Copy files into split_data/train and split_data/test ----
    if OUTPUT_DIR.exists():
        print(f"\n{OUTPUT_DIR} already exists — files will be overwritten/merged.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    copy_sequences(train_seq_keys, sequences, "train")
    copy_sequences(test_seq_keys, sequences, "test")

    # ---- Save the frozen split assignment (reproducibility, like fold_assignments.json) ----
    split_record = {
        "grouping_key": "base_hexcode (original + all augmentations grouped together)",
        "test_size": TEST_SIZE,
        "seed": RANDOM_SEED,
        "train_groups": sorted(train_hexes),
        "test_groups": sorted(test_hexes),
    }
    split_path = OUTPUT_DIR / "split_assignments.json"
    with open(split_path, "w") as fp:
        json.dump(split_record, fp, indent=2)

    print(f"\nDone. Split written to: {OUTPUT_DIR.resolve()}")
    print(f"Frozen split assignment saved to: {split_path.resolve()}")


if __name__ == "__main__":
    main()