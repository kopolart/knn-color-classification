"""
simple_demo.py
--------------
A self-contained, beginner-friendly Two-Stage Color Classification demo.

This script does NOT depend on prepare_dataset.py or model.pkl.
It loads the dataset, trains a KNN inline, and demonstrates two-stage prediction.

Dataset:
    KopolArt Color Collection — https://kopolart.ir/colors
    Curated by Ali Yousefi (KopolArt), licensed under CC BY 4.0.
    Attribution: "کپل‌آرت – Ali Yousefi" + https://kopolart.ir/colors

Usage:
    python simple_demo.py                           # run with default test colors
    python simple_demo.py FF1818 388E3C 142F43      # predict specific hex codes

Dataset path:
    Expects 'data/colors.csv' relative to this script's directory,
    or '../data/colors.csv' if running from the simple_version/ folder.

Requirements:
    pip install numpy pandas scikit-learn
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1. Load dataset
#    Try data/colors.csv relative to this file's location first.
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_data_candidates = [
    os.path.join(_script_dir, "data", "colors.csv"),
    os.path.join(_script_dir, "..", "data", "colors.csv"),
    "data/colors.csv",
    "../data/colors.csv",
]
data_path = next((p for p in _data_candidates if os.path.exists(p)), None)
if data_path is None:
    raise FileNotFoundError(
        "Could not find colors.csv. Expected at 'data/colors.csv' "
        "relative to this script or its parent directory."
    )

df = pd.read_csv(data_path)

# ---------------------------------------------------------------------------
# 2. Feature preparation (plain RGB — easy to understand for presentations)
# ---------------------------------------------------------------------------
FEATURES = ['Red (8 bit)', 'Green (8 bit)', 'Blue (8 bit)']
X = df[FEATURES].values
y = df['Base_English'].values

# ---------------------------------------------------------------------------
# 3. Train/test split (80/20, fixed seed for reproducibility)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index.values, test_size=0.20, random_state=42
)

# ---------------------------------------------------------------------------
# 4. Feature scaling & Stage-1 KNN training
#    K=15 is used here (larger K → smoother decision boundary, better for
#    presentations where you want stable results without CV tuning)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
knn.fit(X_train_scaled, y_train)

# Keep the training subset for Stage-2 nearest-neighbor retrieval
train_df = df.iloc[idx_train].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helper: parse a hex code to [R, G, B]
# ---------------------------------------------------------------------------
def hex_to_rgb(hex_code: str) -> list:
    """Convert '#RRGGBB' to [R, G, B] integers."""
    h = hex_code.strip().lstrip('#')
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


# ---------------------------------------------------------------------------
# Two-stage prediction
# ---------------------------------------------------------------------------
def predict_color(hex_code: str) -> dict:
    """
    Predict the bilingual color name for a hex code using two stages:
      Stage 1 — KNN on normalized RGB  → base color family
      Stage 2 — Euclidean nearest-neighbor in RGB space within that family
                → exact English + Persian name

    Args:
        hex_code: Hex color code (e.g. '#FF1818' or 'FF1818').

    Returns:
        dict with keys: hex, base, exact_en, exact_fa, distance.
    """
    r, g, b = hex_to_rgb(hex_code)

    # Stage 1: predict base color family
    scaled = scaler.transform([[r, g, b]])
    predicted_base = knn.predict(scaled)[0]

    # Stage 2: find closest shade in training set within the predicted base family
    candidates = train_df[train_df['Base_English'] == predicted_base]
    if candidates.empty:
        candidates = train_df   # fallback: search all training samples

    cand_rgb = candidates[FEATURES].values
    distances = np.linalg.norm(cand_rgb - [r, g, b], axis=1)
    best = candidates.iloc[np.argmin(distances)]

    code = hex_code if hex_code.startswith('#') else f"#{hex_code}"
    return {
        'hex':      code,
        'base':     predicted_base,
        'exact_en': best['EnglishName'],
        'exact_fa': best['Persian Name'],
        'distance': round(float(distances.min()), 2),
    }


# ---------------------------------------------------------------------------
# Comprehensive evaluation (Confusion Matrix + per-class metrics)
# ---------------------------------------------------------------------------
def print_model_evaluation() -> None:
    """
    Print full evaluation metrics on the test set:
      - Overall accuracy and Top-3 accuracy
      - Per-class TP / FP / FN / TN
      - Precision, Sensitivity (Recall), Specificity, F1-Score
      - Macro and weighted averages
      - Confusion matrix (text format, top 8 classes)
    """
    X_test_scaled = scaler.transform(X_test)
    pred_test = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred_test)

    # Top-3 accuracy
    _, indices = knn.kneighbors(X_test_scaled, n_neighbors=3)
    top3_correct = sum(y_test[i] in y_train[indices[i]] for i in range(len(y_test)))
    top3_acc = top3_correct / len(y_test)

    # Per-class One-vs-Rest metrics
    test_classes = sorted(set(y_test) | set(pred_test))
    cm = confusion_matrix(y_test, pred_test, labels=test_classes)

    metrics = []
    for i, cls in enumerate(test_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = int(tp + fn)

        metrics.append({
            'Class': cls, 'Support': support,
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
            'Accuracy':              (tp + tn) / cm.sum(),
            'Precision':             prec,
            'Sensitivity (Recall)':  rec,
            'Specificity':           spec,
            'F1-Score':              f1,
        })

    mdf = pd.DataFrame(metrics)
    active = mdf[mdf['Support'] > 0].copy()

    macro_prec  = active['Precision'].mean()
    macro_sens  = active['Sensitivity (Recall)'].mean()
    macro_spec  = active['Specificity'].mean()
    macro_f1    = active['F1-Score'].mean()

    w = active['Support'] / active['Support'].sum()
    w_prec = (active['Precision'] * w).sum()
    w_sens  = (active['Sensitivity (Recall)'] * w).sum()
    w_spec  = (active['Specificity'] * w).sum()
    w_f1    = (active['F1-Score'] * w).sum()

    sep = "=" * 86
    print(sep)
    print(f"Stage 1 Model Evaluation  (Test set: n={len(y_test)} samples)")
    print(f"   {df['Base_English'].nunique()} base families in dataset | "
          f"{len(active)} present in test | {len(test_classes)} evaluated")
    print(sep)
    print(f"  Overall Accuracy    : {acc * 100:.2f}%  |  Top-3 Accuracy: {top3_acc * 100:.2f}%")
    print(f"  Macro Precision     : {macro_prec * 100:.2f}%  |  Weighted: {w_prec * 100:.2f}%")
    print(f"  Macro Sensitivity   : {macro_sens * 100:.2f}%  |  Weighted: {w_sens * 100:.2f}%")
    print(f"  Macro Specificity   : {macro_spec * 100:.2f}%  |  Weighted: {w_spec * 100:.2f}%")
    print(f"  Macro F1-Score      : {macro_f1 * 100:.2f}%  |  Weighted: {w_f1 * 100:.2f}%")
    print("-" * 86)
    print("  Note: High Specificity (~98%) is expected in multi-class One-vs-Rest evaluation.")
    print("  Note: Weighted Recall equals Overall Accuracy by mathematical identity.")
    print("-" * 86)

    # Per-class table (top 8 by support)
    top8 = active.sort_values('Support', ascending=False).head(8).copy()
    for col in ['Accuracy', 'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1-Score']:
        top8[col] = top8[col].apply(lambda v: f"{v * 100:.1f}%")
    cols = ['Class', 'Support', 'TP', 'FP', 'FN', 'TN',
            'Accuracy', 'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1-Score']
    print("Per-Class Metrics (Top 8 Base Categories by Support):")
    print(top8[cols].to_string(index=False))
    print("-" * 86)

    # Confusion matrix (top 8 classes)
    top8_names = active.sort_values('Support', ascending=False).head(8)['Class'].tolist()
    mask = [t in top8_names for t in y_test]
    cm8  = confusion_matrix(
        np.array(y_test)[mask], np.array(pred_test)[mask], labels=top8_names
    )
    cm_df = pd.DataFrame(
        cm8,
        index   =[c[:6] for c in top8_names],
        columns =[c[:6] for c in top8_names],
    )
    print("Confusion Matrix (Top 8 Classes — Rows: Actual, Columns: Predicted):")
    print(cm_df.to_string())
    print(sep)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    DEFAULT_COLORS = ['#FF1818', '#F4722B', '#EAC100', '#388E3C',
                      '#10375C', '#7C3E66', '#FAEEE7']
    test_colors = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_COLORS

    # Print comprehensive evaluation metrics
    print_model_evaluation()

    # Run two-stage predictions
    print("\n" + "=" * 86)
    print("Two-Stage Color Classification Demo  (KNN Stage 1 + Nearest-Neighbor Stage 2)")
    print("=" * 86)

    for c in test_colors:
        res = predict_color(c)
        print(
            f"Input: {res['hex']:<8}  "
            f"Stage 1 (Base): {res['base']:<14}  "
            f"Stage 2 (Exact): {res['exact_en']} ({res['exact_fa']})  "
            f"[RGB dist: {res['distance']}]"
        )

    print("=" * 86)
    print("Done!")
