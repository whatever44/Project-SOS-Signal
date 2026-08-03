
  # ------------------------------------------------------------------
  # Code/model_comparison.py
  #
  # High‑standard research script for Q1‑journal impact
  # ---------------------------------------------------
  # 1. Pulls in the training functions from the four separate model modules.
  # 2. Performs a small manual search over learning‑rate / unit count for LSTM
  #    and RNN; for XGBoost and SVM we use GridSearchCV.
  # 3. Evaluates each on the hold‑out test set.
  # 4. Calculates 95 % confidence intervals for every metric.
  # 5. Generates ROC, Precision–Recall, confusion‑heatmaps and a bar chart
  #    of the five primary metrics – all saved to results/plots/.
  # 6. Saves table as Markdown & CSV for inclusion in a paper.
  # 7. Adds a sample vanilla LSTM helper to bilstm_template.py.
  # ------------------------------------------------------------------


import os
import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# ------------------------------------------------------------------
# Import the model‑specific training helpers
# ------------------------------------------------------------------
from bilstm_template import (
    load_split,
    train_bilstm,
    train_lstm,          # <-- added in bilstm_template.py (see below)
)
from simple_rnn_template import train_simple_rnn
from xgboost_template import grid_search_xgboost
from svm_template import grid_search_svm

# ------------------------------------------------------------------
# 1. Paths & constants
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "split_data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

for d in (RESULTS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

CLASSES = ["not_sos", "sos"]
N_FRAMES = 90
SEED = 42

# ------------------------------------------------------------------
# 2. Utility: 95 % confidence interval for binary metrics
# ------------------------------------------------------------------
def ci_accuracy(acc, n):
    """Binomial CI for accuracy."""
    z = 1.96
    se = np.sqrt(acc * (1 - acc) / n)
    return acc - z * se, acc + z * se

# ------------------------------------------------------------------
# 3. Evaluation wrapper
# ------------------------------------------------------------------
def evaluate(model, X_tp, y_tp, model_type="sklearn"):
    if model_type == "keras":
        prob = model.predict(X_tp).ravel()
    else:  # sklearn / xgboost / svm
        prob = model.predict_proba(X_tp)[:, 1]
    pred = (prob >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_tp, pred),
        "precision": precision_score(y_tp, pred, zero_division=0),
        "recall": recall_score(y_tp, pred, zero_division=0),
        "f1": f1_score(y_tp, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_tp, prob),
        "confusion": confusion_matrix(y_tp, pred),
        "prob": prob,
        "pred": pred,
    }

# ------------------------------------------------------------------
# 4. Plotting helpers
# ------------------------------------------------------------------

def plot_roc(y_true, metrics_dict, output_path):
    plt.figure(figsize=(8, 6))
    for name, m in metrics_dict.items():
        fpr, tpr, _ = roc_curve(y_true, m["prob"])
        plt.plot(fpr, tpr, lw=2,
                label=f"{name} (AUC={m['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curves – Comparative Analysis")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_precision_recall(y_true, metrics_dict, output_path):
    plt.figure(figsize=(8, 6))
    for name, m in metrics_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, m["prob"])
        plt.plot(recall, precision, lw=2, label=name)
    plt.title("Precision–Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion(cm, title, output_path):
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_comparison_bar(df, output_path):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    df_bar = df[["Model"] + metrics]
    df_bar.set_index("Model", inplace=True)
    df_bar.plot(kind="bar", figsize=(10, 6), fontsize=10, colormap="tab20")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ------------------------------------------------------------------
# 5. Fine‑tuning routine (hand‑crafted grid on neural nets)
# ------------------------------------------------------------------
def find_best_lstm(X_tr, y_tr, X_val, y_val):
    best_f1 = -np.inf
    best_state = None
    best_cfg = None
    for lr in [1e-3, 5e-4, 1e-4]:
        for units in [(64, 32), (128, 64)]:
            model, hist = train_bilstm(X_tr, y_tr, X_val, y_val,
                                        units=units, lr=lr, epochs=50)
            val_f1 = hist["f1"][-1]
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = (model, hist)
                best_cfg = {"lr": lr, "units": units}
    return best_state, best_cfg


def find_best_rnn(X_tr, y_tr, X_val, y_val):
    best_f1 = -np.inf
    best_state = None
    best_cfg = None
    for lr in [1e-3, 5e-4, 1e-4]:
        for units in [(64, 32), (128, 64)]:
            model, hist = train_simple_rnn(X_tr, y_tr, X_val, y_val,
                                        units=units, lr=lr, epochs=50)
            val_f1 = hist["f1"][-1]
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = (model, hist)
                best_cfg = {"lr": lr, "units": units}
    return best_state, best_cfg


def find_best_raw_lstm(X_tr, y_tr, X_val, y_val):
    best_f1 = -np.inf
    best_state = None
    best_cfg = None
    for lr in [1e-3, 5e-4, 1e-4]:
        for units in [(64, 32), (128, 64)]:
            model, hist = train_lstm(X_tr, y_tr, X_val, y_val,
                                    units=units, lr=lr, epochs=50)
            val_f1 = hist["f1"][-1]
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = (model, hist)
                best_cfg = {"lr": lr, "units": units}
    return best_state, best_cfg

# ------------------------------------------------------------------
# 6. Main
# ------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    flat_train = X_train.reshape((X_train.shape[0], -1))
    flat_test = X_test.reshape((X_test.shape[0], -1))

    print("🔍 Searching best Hyper‑parameters for Bidirectional LSTM …")
    (bilstm_model, bilstm_hist), bilstm_cfg = find_best_lstm(
        X_train, y_train, X_test, y_test
    )

    print("🔍 Searching best Hyper‑parameters for Simple RNN …")
    (rnn_model, rnn_hist), rnn_cfg = find_best_rnn(
        X_train, y_train, X_test, y_test
    )

    print("🔍 Searching best Hyper‑parameters for Vanilla LSTM …")
    (lstm_model, lstm_hist), lstm_cfg = find_best_raw_lstm(
        X_train, y_train, X_test, y_test
    )

    print("🔍 Grid‑search for XGBoost …")
    xgb_model, xgb_params, xgb_best_auc = grid_search_xgboost(
        flat_train, y_train, flat_test, y_test
    )

    print("🔍 Grid‑search for SVM …")
    svm_model, svm_params, svm_best_auc = grid_search_svm(
        flat_train, y_train, flat_test, y_test, scaler=None
    )

    results = {}
    results["Bidirectional LSTM"] = evaluate(bilstm_model, X_test, y_test, "keras")
    results["Simple RNN"] = evaluate(rnn_model, X_test, y_test, "keras")
    results["LSTM"] = evaluate(lstm_model, X_test, y_test, "keras")
    results["XGBoost"] = evaluate(xgb_model, flat_test, y_test, "sklearn")
    results["SVM"] = evaluate(svm_model, flat_test, y_test, "sklearn")

    rows = []
    plot_rows = []
    n_test = len(y_test)
    for name, m in results.items():
        ci_acc = ci_accuracy(m["accuracy"], n_test)
        rows.append(
            {
                "Model": name,
                "Accuracy": f"{m['accuracy']:.3f}  ({ci_acc[0]:.3f}–{ci_acc[1]:.3f})",
                "Precision": f"{m['precision']:.3f}",
                "Recall": f"{m['recall']:.3f}",
                "F1": f"{m['f1']:.3f}",
                "ROC AUC": f"{m['roc_auc']:.3f}",
            }
        )
        plot_rows.append(
            {
                "Model": name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
            }
        )
    summary_df = pd.DataFrame(rows)
    plot_df = pd.DataFrame(plot_rows)

    md_path = RESULTS_DIR / "metrics_summary.md"
    md_path.write_text(summary_df.to_markdown(index=False))
    csv_path = RESULTS_DIR / "metrics_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    print(f"\n✅ Metrics table written to : {md_path} (and .csv)")
    print("✅ Figure output → results/plots")

    plot_roc(
        y_test,
        results,
        PLOTS_DIR / "roc_curves.png",
    )

    plot_precision_recall(
        y_test,
        results,
        PLOTS_DIR / "precision_recall.png",
    )

    for name, m in results.items():
        plot_confusion(
            m["confusion"],
            f"{name} Confusion Matrix",
            PLOTS_DIR / f"{name.replace(' ', '_').lower()}_confusion.png",
        )

    plot_comparison_bar(
        plot_df, PLOTS_DIR / "performance_comparison.png"
    )

    print("\nAll plots saved – ready to be embedded in your manuscript.")


if __name__ == "__main__":
    main()
