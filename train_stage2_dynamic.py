import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from huggingface_hub import HfFileSystem
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from config import HF_REPO
from models.stage2_sequences import build_cnn1d, build_bilstm, build_transformer, build_tcn

# --- Your Original Data Loader Snippet (Preserved) ---
def load_hf_dataset_private(repo_id=HF_REPO, token=None):
    # (Insert your provided token here or use env var)
    HF_TOKEN = "" 
    
    fs = HfFileSystem(token=HF_TOKEN)
    base_path = f"datasets/{repo_id}/data"
    
    try:
        fs.ls(base_path)
    except Exception as e:
        print(f"Auth Failed: {e}")
        return None, None

    X_data = []
    y_data = []
    
    classes = {'not_sos': 0, 'sos': 1}
    
    for class_name, label in classes.items():
        class_dir = f"{base_path}/{class_name}"
        
        # Check if class folder exists
        if not fs.exists(class_dir):
            print(f"Warning: Folder {class_dir} not found. Skipping.")
            continue
            
        # Get all sample folders
        sample_folders = fs.ls(class_dir, detail=False)
        
        print(f"Processing {class_name} ({len(sample_folders)} samples)...")
        
        for i, sample_path in enumerate(sample_folders):
            landmarks_path = f"{sample_path}/landmarks"
            print(i)
            if not fs.exists(landmarks_path):
                continue
            
            # Use glob to find .npy files
            npy_files = fs.glob(f"{landmarks_path}/*.npy")
            npy_files = sorted(npy_files) # Critical to keep time order
            
            # Strict check: We need exactly 90 frames
            if len(npy_files) != 90:
                continue
                
            sample_sequence = []
            
            for npy_file in npy_files:
                # 'fs.open' works like standard python open, but over the web
                with fs.open(npy_file, 'rb') as f:
                    frame_data = np.load(f)
                    frame_data = frame_data.flatten()
                    
                    # Pad if single hand (63 -> 126)
                    if frame_data.shape[0] == 63:
                        frame_data = np.pad(frame_data, (0, 63), 'constant')
                        
                    sample_sequence.append(frame_data)
            
            X_data.append(np.array(sample_sequence))
            y_data.append(label)
            
            # Progress bar
            if i > 0 and i % 50 == 0:
                print(f"  Loaded {i} samples...", end='\r')

    X = np.array(X_data)
    y = np.array(y_data)
    
    print(f"\nFinal Dataset Shape: {X.shape}")
    return X, y

def train_stage2():
    print("=== Loading Stage 2 Data ===")
    X, y = load_hf_dataset_private()
    if X is None: return

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = (90, 126) # (Time, Features)

    # List of Architectures to Grid Search
    architectures = {
        'CNN_1D': build_cnn1d,
        'BiLSTM': build_bilstm,
        'Transformer': build_transformer,
        'TCN': build_tcn
    }
    
    # Training Loop
    for name, builder in architectures.items():
        print(f"\n=== Training Architecture: {name} ===")
        
        model = builder(input_shape)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks for Fine-Tuning
        callbacks = [
            # Reduce LR if stuck (Finetuning on the fly)
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            # Stop early to prevent overfitting
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
            # Save best checkpoint
            ModelCheckpoint(f'saved_models/stage2_{name}.keras', save_best_only=True)
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=30, # Grid search usually uses fewer epochs, increase for final
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Final Eval
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Results {name} -> Accuracy: {acc:.4f}, Loss: {loss:.4f}")

if __name__ == "__main__":
    import os
    os.makedirs('saved_models', exist_ok=True)
    train_stage2()