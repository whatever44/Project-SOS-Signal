import os
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "split_data"
MODEL_DIR = DATA_DIR = Path(__file__).resolve().parent.parent / "models"
CLASSES = ["not_sos", "sos"]
N_FRAMES = 90


def load_split(split_name):
    folder_root = os.path.join(DATA_DIR, split_name)
    sequences = {}
    labels = {}

    for label, class_name in enumerate(CLASSES):
        folder = os.path.join(folder_root, class_name)
        for fname in os.listdir(folder):
            if not fname.endswith(".npy"):
                continue
            cls, hex_id, frame_num = fname[:-4].split("__")
            frame_num = int(frame_num)
            key = class_name + "_" + hex_id
            if key not in sequences:
                sequences[key] = {}
                labels[key] = label
            arr = np.load(os.path.join(folder, fname))
            arr_flat = arr.flatten()
            if arr_flat.size < 126:
                arr_flat = np.pad(arr_flat, (0, 126 - arr_flat.size), 'constant')
            sequences[key][frame_num] = arr_flat

    X, y = [], []
    for key, frame_dict in sequences.items():
        frames = [frame_dict[i] for i in sorted(frame_dict.keys())]
        if len(frames) != N_FRAMES:
            print(f"Warning: Expected {N_FRAMES} frames but got {len(frames)} for sample {key}. Skipping.")
            continue
        X.append(np.stack(frames))
        y.append(labels[key])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y


if __name__ == "__main__":
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    # Flatten sequences for XGBoost: (samples, n_frames * feature_dim)
    feature_dim = X_train.shape[2]
    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_test_flat = X_test.reshape((X_test.shape[0], -1))

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
    )

    model.fit(X_train_flat, y_train)

    model_path = MODEL_DIR / "xgboost_sos.model"
    joblib.dump(model, str(model_path))
    print(f"XGBoost model saved to {model_path}")