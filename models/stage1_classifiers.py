from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import numpy as np

class HeuristicSOSClassifier:
    """
    A rule-based classifier. In production CV, we often use this 
    as a 'sanity check' or fallback if the ML model has low confidence.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        # Example Heuristic: If thumb is tucked inside fingers (Geometric check)
        # Expects flattened landmarks. This is a simplified placeholder logic.
        preds = []
        for sample in X:
            # Reshape to (21, 3) for the first hand
            # Assuming first 63 features are hand 1
            hand = sample[:63].reshape(21, 3)
            
            # Simple Logic: Is Thumb Tip (4) to the right of Index Base (5)? (Right hand specific)
            # In a real scenario, you calculate Euclidean distances between fingertips and palm
            thumb_tip = hand[4]
            index_base = hand[5]
            
            # This is a dummy rule for demonstration
            if thumb_tip[0] < index_base[0]: 
                preds.append(1) # Potential Trigger
            else:
                preds.append(0)
        return np.array(preds)

def get_stage1_model_grid():
    """
    Returns dictionary of models and their hyperparam grids.
    """
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            # SOTA for Tabular Data
            'model': XGBClassifier(eval_metric='logloss', random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6]
            }
        },
        'SVM': {
            # Traditional Geometric Boundary
            'model': SVC(probability=True, random_state=42, class_weight='balanced'),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'poly']
            }
        },
        'KNN': {
            # Distance-based baseline
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        }
    }
    return models