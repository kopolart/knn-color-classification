"""
predict.py
----------
Two-Stage Color Inference: load 'model.pkl' and predict the bilingual color name
for any hex code.

Stage 1 — Base Family (KNN):
    Uses the trained knn_base_model to classify the input color into a coarse
    color family (e.g. 'Royal Blue').

Stage 2 — Exact Shade (CIE Lab Delta-E nearest-neighbor):
    Restricts the search to training samples in the predicted base family, then
    picks the sample with the smallest perceptual distance (Delta-E) in CIE L*a*b*
    space. This keeps the returned bilingual name consistent with the Stage 1
    family; if Stage 1 is wrong, that error carries into Stage 2.

Dataset:
    KopolArt Color Collection — https://kopolart.ir/colors
    Curated by Ali Yousefi (KopolArt), licensed under CC BY 4.0.
    Attribution: "کپل‌آرت – Ali Yousefi" + https://kopolart.ir/colors

Usage:
    python predict.py                        # run the built-in demo set
    python predict.py FF1818 388E3C           # predict specific hex codes

Requirements:
    model.pkl must exist (run train.py first).
    pip install numpy pandas joblib
"""

import sys
import os
import colorsys

import joblib
import numpy as np

# Resolve paths relative to this file so the script works from any directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Shared utilities from prepare_dataset.py
from prepare_dataset import hex_to_rgb, rgb_to_lab

# Ensure UTF-8 output on Windows consoles (Python 3.7+)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def hex_to_features(h: str) -> list:
    """
    Compute the 6-dimensional cylindrical HSV + normalized RGB feature vector.

    This encoding avoids the Hue discontinuity at 360°/0° and suppresses
    hue noise for neutral colors (S ≈ 0 → circular components ≈ 0).

    Args:
        h: Hex color code string (e.g. '#5463FF').

    Returns:
        [S·cos(2πH), S·sin(2πH), V, R_norm, G_norm, B_norm]
    """
    r, g, b = hex_to_rgb(h)
    r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
    hh, ss, vv = colorsys.rgb_to_hsv(r_n, g_n, b_n)
    return [
        ss * np.cos(2 * np.pi * hh),
        ss * np.sin(2 * np.pi * hh),
        vv,
        r_n,
        g_n,
        b_n,
    ]


# ---------------------------------------------------------------------------
# Load the serialized pipeline
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. "
        "Please run 'python train.py' first to train and save the model."
    )

saved          = joblib.load(MODEL_PATH)
knn_base_model = saved["knn_base_model"]
scaler         = saved["scaler"]
train_df       = saved["train_df"]
labs_train     = saved["labs_train"]
base_en_to_fa  = saved.get("base_en_to_fa", {})
en_to_fa       = saved.get("en_to_fa", {})

# ---------------------------------------------------------------------------
# Default demo palette — covers warm, cool, and neutral color ranges
# ---------------------------------------------------------------------------
DEFAULT_TESTS = [
    # Warm colors
    "#FF1818",   # Bright Red
    "#F4722B",   # Dark Orange
    "#EAC100",   # Amber Yellow
    "#FF5F7E",   # Salmon Pink
    "#7C3E66",   # Plum Purple
    "#8B4513",   # Chocolate Brown
    "#FFC090",   # Peach
    # Cool colors
    "#142F43",   # Dark Navy Blue
    "#87CEEB",   # Sky Blue
    "#5463FF",   # Bright Royal Blue
    "#388E3C",   # Dark Green
    "#24A19C",   # Teal
    # Neutral colors
    "#FAEEE7",   # Chalk White
    "#ECECEC",   # Light Gray
    "#1B1B1B",   # Charcoal Black
]

# ---------------------------------------------------------------------------
# Parse command-line hex codes, or fall back to the default list
# ---------------------------------------------------------------------------
if len(sys.argv) > 1:
    parsed = []
    for arg in sys.argv[1:]:
        code = arg.strip()
        code = code if code.startswith('#') else f"#{code}"
        if len(code) == 7:
            parsed.append(code)
        else:
            print(f"  [Warning] Skipping invalid hex code: '{arg}'", file=sys.stderr)
    test_colors = parsed if parsed else DEFAULT_TESTS
else:
    test_colors = DEFAULT_TESTS

# ---------------------------------------------------------------------------
# Two-Stage Prediction
# ---------------------------------------------------------------------------
print("=" * 80)
print("Two-Stage Color Prediction")
print("   Stage 1: Base family  ->  KNN on cylindrical HSV + RGB features")
print("   Stage 2: Exact shade  ->  CIE Lab Delta-E nearest-neighbor retrieval")
print("=" * 80)

for hex_code in test_colors:
    # --- Input validation ---
    try:
        r, g, b = hex_to_rgb(hex_code)
    except ValueError as err:
        print(f"[Error] {err}")
        continue

    # --- Feature extraction & scaling ---
    feat = np.array([hex_to_features(hex_code)])
    feat_scaled = scaler.transform(feat)
    query_lab   = np.array(rgb_to_lab(r, g, b))

    # --- Stage 1: predict base color family ---
    pred_base_en = knn_base_model.predict(feat_scaled)[0]
    pred_base_fa = base_en_to_fa.get(pred_base_en, "Other")

    # --- Stage 2: CIE Lab ΔE search within the predicted base family ---
    base_mask = (train_df['Base_English'] == pred_base_en)
    if not np.any(base_mask):
        # Fallback: search the entire training set if the base family is empty
        candidate_sub  = train_df
        candidate_labs = labs_train
    else:
        candidate_sub  = train_df[base_mask]
        candidate_labs = labs_train[base_mask.values]

    diffs         = candidate_labs - query_lab
    distances_lab = np.linalg.norm(diffs, axis=1)
    best_idx      = np.argmin(distances_lab)

    best_row  = candidate_sub.iloc[best_idx]
    full_en   = best_row['EnglishName']
    full_fa   = best_row['Persian Name']
    delta_e   = distances_lab[best_idx]

    # --- Output ---
    print(f"Code : {hex_code}")
    print(f"  Stage 1 (Base family) : {pred_base_en:<20} | {pred_base_fa}")
    print(f"  Stage 2 (Exact shade) : {full_en:<20} | {full_fa}  (Delta-E = {delta_e:.2f})")
    print("-" * 80)
