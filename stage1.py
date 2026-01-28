import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# -------------------------------
# FEATURE ENGINEERING (RATIOS)
# -------------------------------
def extract_ratio_features(landmarks):
    """
    landmarks: (63,) -> 21 points (x,y,z)
    Returns ~10 ratio/angle features
    """

    pts = landmarks.reshape(21, 3)

    def dist(a, b):
        return np.linalg.norm(pts[a][:2] - pts[b][:2])

    palm = dist(0, 9) + 1e-6

    features = [
        dist(4, 8) / palm,     # thumb-index
        dist(8, 12) / palm,    # index-middle
        dist(12, 16) / palm,   # middle-ring
        dist(16, 20) / palm,   # ring-pinky
        dist(4, 20) / palm,    # thumb-pinky
        dist(0, 8) / palm,     # palm-index
        dist(0, 12) / palm,    # palm-middle
        dist(0, 16) / palm,    # palm-ring
        dist(0, 20) / palm,    # palm-pinky
        dist(5, 17) / palm     # palm width
    ]

    return np.array(features)


def train_stage_1_balanced(csv_path):

    # 1. Load Data
    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if 'hand_mark' in c]
    X_raw = df[feature_cols].values
    y = df['sign'].values

    # Trigger = class 7
    y_binary = (y == 7).astype(int)

    print(f"Original Class Distribution:\n{pd.Series(y_binary).value_counts()}")

    # 2. Extract Ratio Features
    X_features = np.array([extract_ratio_features(row) for row in X_raw])

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    # 4. SMOTE on training only
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Resampled Training Distribution:\n{pd.Series(y_train_resampled).value_counts()}")

    # 5. Pipeline: Scaling + Linear SVM
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(class_weight="balanced", max_iter=5000))
    ])

    model.fit(X_train_resampled, y_train_resampled)

    # 6. Evaluation
    y_pred = model.predict(X_test)

    print("\n--- Stage 1 Classification Report (Linear SVM) ---")
    print(classification_report(y_test, y_pred, target_names=['Other', 'Trigger']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Other', 'Trigger'],
                yticklabels=['Other', 'Trigger'])
    plt.title('Stage 1: Confusion Matrix (Linear SVM)')
    plt.show()

    # 7. Save
    joblib.dump(model, 'stage1_linear_svm_trigger.pkl')
    print("Stage 1 Linear SVM model saved.")


# Usage
train_stage_1_balanced('hand_landmarks.csv')
