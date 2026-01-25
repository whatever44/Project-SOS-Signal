import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Ensure these match your actual folder structure
from config import CSV_PATH, RANDOM_SEED, CV_FOLDS
from models.stage1_classifiers import get_stage1_model_grid, HeuristicSOSClassifier

def train_stage1():
    print("=== Loading Data ===")
    df = pd.read_csv(CSV_PATH)
    
    # 1. Feature Selection
    feature_cols = [c for c in df.columns if 'hand_mark' in c]
    X = df[feature_cols].values
    
    # 2. Binarize Labels (CRITICAL CORRECTION)
    # Class 7 is the 'Trigger' (SOS), everything else (1-6) is 'Non-SOS'
    # 7 becomes 1, everything else becomes 0
    y_raw = df['sign'].values
    y = (y_raw == 7).astype(int)
    
    print("Class Distribution (0=Non-SOS, 1=SOS):")
    print(pd.Series(y).value_counts())
    
    # 3. Split
    # Stratify is crucial here to ensure we have enough 'SOS' samples in the test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # 4. Get Models
    model_grid = get_stage1_model_grid()
    
    best_overall_score = 0
    best_overall_model = None

    print("\n=== Starting Grid Search (Binary Classification with SMOTE) ===")
    
    for name, config in model_grid.items():
        print(f"\nTraining {name}...")
        
        # PIPELINE: Scaling -> SMOTE -> Model
        # SMOTE generates synthetic 'SOS' (Class 1) samples ONLY during training
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=RANDOM_SEED)),
            ('classifier', config['model'])
        ])
        
        # Adjust params for pipeline format
        pipe_params = {f'classifier__{k}': v for k, v in config['params'].items()}
        
        # Scoring changed to 'f1' to prioritize the Positive Class (SOS) performance
        clf = GridSearchCV(pipeline, pipe_params, cv=CV_FOLDS, scoring='f1', n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # Evaluate
        score = clf.best_score_
        print(f"  Best CV F1 (SOS Class): {score:.4f}")
        print(f"  Best Params: {clf.best_params_}")
        
        # Test Set Evaluation
        test_pred = clf.predict(X_test)
        print(classification_report(y_test, test_pred, target_names=['Non-SOS', 'SOS']))
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_model = clf.best_estimator_

    # 5. Save Best Model
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(best_overall_model, 'saved_models/stage1_best_ml.pkl')
    print(f"\nSaved best model to saved_models/stage1_best_ml.pkl")

    # --- Heuristic Sanity Check ---
    print("\n=== Running Heuristic Check ===")
    heuristic = HeuristicSOSClassifier()
    
    # Run prediction on test set
    h_preds = heuristic.predict(X_test)
    
    print("Heuristic Report (Rule-Based):")
    print(classification_report(y_test, h_preds, target_names=['Non-SOS', 'SOS']))

if __name__ == "__main__":
    train_stage1()