import os
import numpy as np

TARGET_FRAMES = 90

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input
DATA_FLAT_DIR = os.path.join(BASE_DIR, "data", "data_flat (2)", "data_flat")

# Output
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "zero_padding_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def zero_pad(sequence, target_frames=90):
    """Pad a sequence with zeros until it has target_frames."""
    current_frames = sequence.shape[0]

    if current_frames >= target_frames:
        return sequence

    pad_size = target_frames - current_frames
    padding = np.zeros((pad_size, *sequence.shape[1:]), dtype=sequence.dtype)

    return np.concatenate((sequence, padding), axis=0)


for file in os.listdir(DATA_FLAT_DIR):

    if not file.endswith(".npy"):
        continue

    input_path = os.path.join(DATA_FLAT_DIR, file)
    sequence = np.load(input_path)
    padded = zero_pad(sequence, TARGET_FRAMES)

    output_path = os.path.join(OUTPUT_DIR, file)
    np.save(output_path, padded)

    print(f"{file}: {sequence.shape} -> {padded.shape}")

print("\n✅ All files processed successfully!")