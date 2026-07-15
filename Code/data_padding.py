import os
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration for the padding process
# ---------------------------------------------------------------------------
# The script expects a base data directory containing sequences where each
# sample is stored as ``{prefix}__{frame_index}.npy``.
# Adjust the following paths if you run the script from a non‑standard
# location.  They resolve relative to the repository root.
INPUT_DIR = Path("../Project-SOS-Signal/data")  # base data folder
OUTPUT_DIR = Path("../Project-SOS-Signal/data/padded_data")  # destination for padded sequences
# Desired sequence length – the training scripts expect exactly this many
# frames per sample.
TARGET_FRAMES = 90
# If ``True`` and a sequence has more than TARGET_FRAMES frames, only the
# first TARGET_FRAMES images are retained.  If ``False`` the full sequence
# is written verbatim and zero‑frame padding is appended to reach the
# target length.
TRUNCATE_IF_LONGER = False
# ----------------------------------------------------------------------------------------

FRAME_PATTERN = re.compile(r"^(.*)__(\d+)\.npy$")


def find_npy_files(root: Path):
    """Recursively find all .npy files under the given root.

    Parameters
    ----------
    root: Path
        Directory to search starting from.

    Returns
    -------
    List[Path]
        A list of paths to all matching .npy files, sorted by os.walk order.

    Notes
    -----
    The function uses :py:meth:`Path.rglob` which searches depth‑first; the order
    is deterministic but not guaranteed to be chronological.  The caller is
    responsible for filtering duplicates or ignoring files that do not match the
    expected naming scheme.
    """
    return list(root.rglob("*.npy"))


def group_sequences(files):
    """Group individual .npy files into per‑sequence mapping.

    Each hand gesture sample is stored as a set of 90 or more ``frame_idx``
    files that share a common *prefix* (e.g. ``sample_001``).  This helper
    rolls those files into a dictionary

    ``{ (relative_dir, prefix) : {frame_index: Path, ...} }``

    Parameters
    ----------
    files: Iterable[Path]
        Paths returned by :func:`find_npy_files`.

    Returns
    -------
    defaultdict[tuple, dict]
        A dict mapping a (relative directory, prefix) tuple to a mapping of
        frame indices -> paths.

    Notes
    -----
    The function prints a warning if a file does not match the expected
    ``name__idx.npy`` pattern.  It also keeps track of the relative directory
    from :data:`INPUT_DIR` so that the output structure mirrors the input
    structure.
    """
    groups = defaultdict(dict)
    for f in files:
        match = FRAME_PATTERN.match(f.name)
        if not match:
            print(f"Skipping file with unexpected name format: {f.name}")
            continue
        prefix, frame_idx = match.group(1), int(match.group(2))
        rel_dir = f.parent.relative_to(INPUT_DIR)
        key = (rel_dir, prefix)
        groups[key][frame_idx] = f
    return groups


def pad_sequence(rel_dir, prefix, frame_map):
    """Pad or truncate a sequence to :data:`TARGET_FRAMES`.

    The input ``frame_map`` is a mapping of frame index to the full path of
    the corresponding ``.npy`` file.  The function writes a new directory
    `<OUT_DIR>/<rel_dir>` containing exactly ``TARGET_FRAMES`` files for the
    given ``prefix``.

    Padding strategy
    ----------------
    1. If the original sequence has more than ``TARGET_FRAMES`` frames and
       :data:`TRUNCATE_IF_LONGER` is ``True``, it simply keeps the first
       ``TARGET_FRAMES`` indices.
    2. If truncation is disabled, the original frames are written verbatim
       and then additional zero vectors are appended until the count reaches
       ``TARGET_FRAMES``.

    All frames are written with the same shape/dtype as the original
    sequence.  The :func:`numpy.load` call uses clobber‑protected data to
    avoid accidental memory leaks.

    Parameters
    ----------
    rel_dir: Path
        Directory relative to :data:`INPUT_DIR` where the sequence resides.
    prefix: str
        Base name of the sequence (excluding the ``__<idx>`` part).
    frame_map: Dict[int, Path]
        Mapping of integer frame index to the full path to the .npy file.

    Returns
    -------
    None
        Side effects only – writes the padded files to :data:`OUTPUT_DIR`.
    """
    out_dir = OUTPUT_DIR / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_indices = sorted(frame_map.keys())
    num_existing = len(existing_indices)

    expected = list(range(num_existing))
    if existing_indices != expected:
        print(f"Non-contiguous frame indices for {prefix} in {rel_dir}: {existing_indices}")

    sample = np.load(frame_map[existing_indices[0]])
    shape = sample.shape
    dtype = sample.dtype

    if num_existing > TARGET_FRAMES and TRUNCATE_IF_LONGER:
        keep_indices = existing_indices[:TARGET_FRAMES]
    else:
        keep_indices = existing_indices

    for idx in keep_indices:
        data = np.load(frame_map[idx])
        out_path = out_dir / f"{prefix}__{idx}.npy"
        np.save(out_path, data)

    if num_existing < TARGET_FRAMES:
        for idx in range(num_existing, TARGET_FRAMES):
            zero_frame = np.zeros(shape, dtype=dtype)
            out_path = out_dir / f"{prefix}__{idx}.npy"
            np.save(out_path, zero_frame)

    final_count = min(len(keep_indices), TARGET_FRAMES) if TRUNCATE_IF_LONGER else max(num_existing, TARGET_FRAMES)
    if num_existing >= TARGET_FRAMES and not TRUNCATE_IF_LONGER:
        final_count = num_existing
    print(f"{prefix} [{rel_dir}]: {num_existing} → {final_count} frames")


def main():
    """Entry point for the padding script.

    The routine verifies the existence of :data:`INPUT_DIR`, discovers
    all ``.npy`` samples, groups them by sequence, and then writes each
    sequence padded (or truncated) to :data:`OUTPUT_DIR`.

    The console output serves as a lightweight progress display and
    also highlights any irregularities such as missing frames or non‑contiguous
    indices.
    """
    if not INPUT_DIR.exists():
        print(f"Input directory not found: {INPUT_DIR.resolve()}")
        print(f"\nHere's what's actually in the current directory ({Path('.').resolve()}):")
        for p in Path(".").iterdir():
            print(f"  - {p.name}")
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_npy_files(INPUT_DIR)
    print(f"Found {len(files)} .npy files under {INPUT_DIR}")

    groups = group_sequences(files)
    print(f"Grouped into {len(groups)} sequences\n")

    for (rel_dir, prefix), frame_map in groups.items():
        pad_sequence(rel_dir, prefix, frame_map)

    print(f"\nDone. Padded data written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
