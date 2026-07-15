# SOS Signal Gesture Recognition

This repository contains a small utility (`data_padding.py`) that pads or truncates raw hand‑landmark NPY files to a fixed length of **90 frames** – the size expected by the sequence models.  The data and the original training scripts are kept in a legacy directory (`Part_1`), but the official repository layout focuses on the current data processing workflow.

---

## Repository layout

```
Project-SOS-Signal/
├─ .venv/                # Python virtual environment (Python 3.8+)
├─ Code/                 # Current source code
│  └─ data_padding.py     # Pad/truncate .npy gesture sequences
├─ data/                 # Raw gesture samples (a single‑hand, 90‑frame examples)
│  └─ not_sos/
│        └─ not_sos__1766234603_f2a6d95d__0.npy  # (many files – one per frame)
├─ data/padded_data/     # Destination for padded output (created by data_padding.py)
├─ .gitignore            # Exclude large .npy files
├─ README.md              # This file
```

> **Note:** All training scripts (`train_stage1_static.py`, `train_stage2_dynamic.py`, etc.) as well as the trained checkpoints live in the legacy `Part_1/` directory and are *not* part of the current public layout.

---

## Quick‑start guide

1. **Create a virtual environment** (once per platform):
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install required packages**:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost tensorflow huggingface-hub
   ```

3. **Verify the environment** (optional linting):
   ```bash
   flake8 Code
   ```

4. **Pad the raw gesture data**:
   ```bash
   python Code/data_padding.py
   ```
   Per‑sequence, the script writes out exactly 90 frames to `data/padded_data/`.  If a sequence is shorter, zero‑frames are appended; if it is longer, the first 90 frames are kept (you can force truncation by setting `TRUNCATE_IF_LONGER = True` in the script).

---

## Data description

| Path | Shape / Content | Notes |
|------|-----------------|-------|
| `data/not_sos/` | `{prefix}__<frame>.npy` | Each 1‑D `float32` array of shape `(63,)` (single‑hand).  All samples are expected to contain **90 frames**. |
| `data/padded_data/` | Same as above, but always 90 frames. | Output of `data_padding.py`. |

The raw `data/` folder is a copy of the public portion of the original dataset.  The original private dataset is downloaded via the training scripts in `Part_1`.

---

## `data_padding.py` – key points

* **Purpose** – Convert arbitrary‑length NPY gesture files into a strict 90‑frame sequence.
* **Configuration** – Edit the global constants at the top of the file to change the input/output directories, target length, and truncation policy.
* **Output** – For each sequence, a subdirectory mirroring the input structure is created under `data/padded_data/`.
* **Usage** – The script is a self‑contained CLI: simply run `python Code/data_padding.py`.

---

## Extending the workflow

* To add new preprocessing steps, modify the logic inside `pad_sequence`.
* For large datasets, consider streaming the NPY files instead of loading the full sequence into memory.
* Unit tests can be added under a `tests/` folder and run with `pytest`.

---

## Other notes

* The `Part_1/` folder houses the legacy training scripts and model checkpoints.  If you need to run the full training pipeline, refer to the legacy scripts; they have **not** been migrated to the current root directory.
* The repository intentionally keeps the training code separate to avoid cluttering the public-facing data‑processing utilities.

---

Happy coding!