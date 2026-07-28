import os
import datetime
import numpy as np
import tensorflow as tf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "split_data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
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
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    feature_dim = X_train.shape[2]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_FRAMES, feature_dim)),
        tf.keras.layers.SimpleRNN(64, return_sequences=True),
        tf.keras.layers.SimpleRNN(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    log_dir = Path("logs") / f"rnn_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    try:
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1)
    except Exception as e:
        print(f"TensorBoard not available ({e}). Using simple logging callback.")
        class DummyTensorBoard(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                print(f"Epoch {epoch + 1} - loss: {logs.get('loss'):.4f}, accuracy: {logs.get('accuracy'):.4f}")
        tensorboard_cb = DummyTensorBoard()

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[tensorboard_cb],
    )

    model_path = MODEL_DIR / "simple_rnn_sos_model.keras"
    model.save(model_path)
    print(f"Simple RNN model saved to {model_path}")