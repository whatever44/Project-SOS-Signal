import os
import numpy as np

OUTPUT_DIR = "/Users/sitasma/Project-SOS-Signal/Part_2/data/padded_data"

all_good = True

for file in os.listdir(OUTPUT_DIR):
    if not file.endswith(".npy"):
        continue
    seq = np.load(os.path.join(OUTPUT_DIR, file))
    status = "✅" if seq.shape[0] == 90 else "❌"
    if seq.shape[0] != 90:
        all_good = False
    print(f"{status} {file}: {seq.shape}")

print("\n✅ All files are 90 frames!" if all_good else "\n❌ Some files are NOT 90 frames!")