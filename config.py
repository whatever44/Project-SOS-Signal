import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'stage1_csv', 'hand_landmarks.csv')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Stage 2 Data
HF_REPO = "MIssion-Ctrl/SOS-Gesture"
# SEQUENCE_LENGTH = 90  # Defined in your snippet
# INPUT_DIM = 126       # 2 hands * 21 points * 3 coords

# Hyperparams
RANDOM_SEED = 42
CV_FOLDS = 3