"""
prepare_dataset.py
------------------
Transforms the raw color dataset ('data/colors-raw.csv') into the feature-rich
'data/colors.csv' used for training.

Dataset:
    KopolArt Color Collection — https://kopolart.ir/colors
    Curated by Ali Yousefi (KopolArt), licensed under CC BY 4.0.
    Attribution: "کپل‌آرت – Ali Yousefi" + https://kopolart.ir/colors

Two-Stage Color Classification preparation includes:

  1. Modifier Removal & Compound Exception Handling:
     - Strips true modifiers (dark, light, pale, deep, bright, vivid, dim, medium, etc.)
     - Preserves compound names (Sky Blue, Navy Blue, Petrol Blue, Royal Blue, Olive Green, etc.)
     - Resolves prefix/suffix variations (Dark Navy, Blue Navy, Dark Navy Blue -> Navy Blue / سرمه‌ای)

  2. Color Space Conversions:
     - 8-bit RGB integers (Red, Green, Blue in [0, 255])
     - Hue angle (0–360 degrees)
     - HSL Saturation and Lightness (%)
     - HSV Saturation and Value (%)
     - Perceptually-uniform CIE L*a*b* coordinates (L*, a*, b*) — Euclidean distance
       in this space corresponds to human perceptual difference (Delta E / ΔE).

  3. Base Color Family extraction in English and Persian.

Usage:
    python prepare_dataset.py          # reads data/colors-raw.csv, writes data/colors.csv

Requirements:
    pip install numpy pandas
"""

import os
import re
import colorsys
import numpy as np
import pandas as pd


# Resolve the repository root so the script works from any working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# True modifiers to strip from color names (prefixes / adjectives)
# These words add nuance but don't change the base color family.
# ---------------------------------------------------------------------------
MODIFIERS = {
    'dark', 'light', 'pale', 'deep', 'bright', 'vivid', 'dim', 'medium',
    'soft', 'dull', 'faded', 'smoky', 'smoke', 'icy', 'warm', 'cool',
    'rich', 'pure', 'intense', 'very', 'ultra', 'mellow', 'dusty',
    'faint', 'subtle', 'neon', 'heavy', 'pastel', 'clear'
}

# ---------------------------------------------------------------------------
# Compound color names and their canonical (Base_English, Base_Persian) pairs.
# Keys are lowercased full or partial color names.
# Order matters: more-specific entries should appear before generic ones,
# but since Python dicts are ordered, we just check `clean` first then
# the stripped candidate — so specificity is handled by the exact-match lookup.
# ---------------------------------------------------------------------------
COMPOUND_MAP = {
    # ── Navy variants ──────────────────────────────────────────────────────
    'dark navy':       ('Navy Blue', 'سرمه‌ای'),
    'blue navy':       ('Navy Blue', 'سرمه‌ای'),
    'navy':            ('Navy Blue', 'سرمه‌ای'),
    'navy blue':       ('Navy Blue', 'سرمه‌ای'),
    'dark navy blue':  ('Navy Blue', 'سرمه‌ای'),
    'light navy blue': ('Navy Blue', 'سرمه‌ای'),
    'charcoal navy':   ('Navy Blue', 'سرمه‌ای'),

    # ── Sky Blue ───────────────────────────────────────────────────────────
    'sky blue':        ('Sky Blue', 'آبی آسمانی'),
    'light sky blue':  ('Sky Blue', 'آبی آسمانی'),
    'dark sky blue':   ('Sky Blue', 'آبی آسمانی'),
    'blue sky':        ('Sky Blue', 'آبی آسمانی'),

    # ── Royal Blue ─────────────────────────────────────────────────────────
    'royal blue':        ('Royal Blue', 'آبی کاربنی'),
    'bright royal blue': ('Royal Blue', 'آبی کاربنی'),

    # ── Petrol Blue ────────────────────────────────────────────────────────
    'petrol blue':      ('Petrol Blue', 'آبی نفتی'),
    'dark petrol blue': ('Petrol Blue', 'آبی نفتی'),
    'muted petrol blue':('Petrol Blue', 'آبی نفتی'),

    # ── Mint Green ─────────────────────────────────────────────────────────
    'mint green':      ('Mint Green', 'سبز نعنایی'),
    'dark mint green': ('Mint Green', 'سبز نعنایی'),
    'pale mint':       ('Mint Green', 'سبز نعنایی'),

    # ── Olive Green ────────────────────────────────────────────────────────
    'olive green':          ('Olive Green', 'سبز زیتونی'),
    'deep olive espresso':  ('Olive Green', 'سبز زیتونی'),
    'olive charcoal':       ('Olive Green', 'سبز زیتونی'),

    # ── Army Green ─────────────────────────────────────────────────────────
    'army green':      ('Army Green', 'سبز ارتشی'),
    'deep army green': ('Army Green', 'سبز ارتشی'),

    # ── Mustard Yellow ─────────────────────────────────────────────────────
    'mustard yellow':      ('Mustard Yellow', 'زرد خردلی'),
    'dark mustard yellow': ('Mustard Yellow', 'زرد خردلی'),
    'dark mustard':        ('Mustard Yellow', 'زرد خردلی'),

    # ── Golden Yellow ──────────────────────────────────────────────────────
    'golden yellow': ('Golden Yellow', 'زرد طلایی'),
    'gold yellow':   ('Golden Yellow', 'زرد طلایی'),

    # ── Coral Red ──────────────────────────────────────────────────────────
    'coral red':    ('Coral Red', 'قرمز مرجانی'),
    'coral orange': ('Coral Red', 'قرمز مرجانی'),
    'coral':        ('Coral Red', 'قرمز مرجانی'),

    # ── Salmon Pink ────────────────────────────────────────────────────────
    'salmon pink': ('Salmon Pink', 'صورتی گلبهی'),
    'salmon':      ('Salmon Pink', 'صورتی گلبهی'),

    # ── Coffee / Chocolate Brown ───────────────────────────────────────────
    'coffee brown':    ('Coffee Brown', 'قهوه‌ای نسکافه‌ای'),
    'coffee bean':     ('Coffee Brown', 'قهوه‌ای نسکافه‌ای'),
    'chocolate brown': ('Chocolate Brown', 'قهوه‌ای شکلاتی'),

    # ── Charcoal variations ────────────────────────────────────────────────
    'charcoal black': ('Charcoal Black', 'مشکی زغالی'),
    'charcoal gray':  ('Charcoal Gray', 'خاکستری زغالی'),

    # ── White variations ───────────────────────────────────────────────────
    'chalk white': ('Chalk White', 'سفید گچی'),
    'bone white':  ('Bone White', 'سفید استخوانی'),
    'milk white':  ('Milk White', 'سفید شیری'),
}

# ---------------------------------------------------------------------------
# Regex patterns for general base-category detection when compound map misses.
# Ordered from most-specific to most-general so the first match wins.
# ---------------------------------------------------------------------------
BASE_FALLBACK_PATTERNS = [
    ('Red',    'قرمز',      r'\b(red|crimson|ruby|scarlet|burgundy|maroon|rose|cherry|wine|cerise)\b'),
    ('Blue',   'آبی',       r'\b(blue|cyan|azure|indigo|sapphire|cerulean|cobalt|denim)\b'),
    ('Teal',   'سبزآبی',    r'\b(teal|turquoise|aqua)\b'),
    ('Green',  'سبز',       r'\b(green|emerald|lime|sage|jade|forest|moss|pistachio)\b'),
    ('Yellow', 'زرد',       r'\b(yellow|amber|lemon|mustard|gold|saffron)\b'),
    ('Orange', 'نارنجی',    r'\b(orange|apricot|tangerine|rust)\b'),
    ('Peach',  'هلویی',     r'\b(peach)\b'),
    ('Pink',   'صورتی',     r'\b(pink|magenta|fuchsia|blush|flamingo)\b'),
    ('Purple', 'بنفش',      r'\b(purple|violet|lavender|lilac|plum|mauve|amethyst|aubergine|eggplant)\b'),
    ('Brown',  'قهوه‌ای',   r'\b(brown|caramel|tan|mocha|chestnut|copper|bronze|cocoa|henna|camel|cinnamon|jujube)\b'),
    ('Gray',   'خاکستری',   r'\b(gray|grey|silver|charcoal|ash|slate|smoke)\b'),
    ('Black',  'مشکی',      r'\b(black|ebony|onyx|jet)\b'),
    ('White',  'سفید',      r'\b(white|snow|chalk|ivory|pearl|icy|seashell|cloud)\b'),
    ('Cream',  'کرمی',      r'\b(cream|vanilla|milk)\b'),
    ('Beige',  'بژ',        r'\b(beige|khaki|nude|sand|sandy|almond|wheat|bone|champagne|honey)\b'),
]

# ---------------------------------------------------------------------------
# English → Persian mapping for base color families.
# Used by other modules (train.py, predict.py) via `from prepare_dataset import EN_TO_FA_BASE`.
# ---------------------------------------------------------------------------
EN_TO_FA_BASE = {
    'Red':         'قرمز',
    'Blue':        'آبی',
    'Green':       'سبز',
    'Yellow':      'زرد',
    'Orange':      'نارنجی',
    'Pink':        'صورتی',
    'Purple':      'بنفش',
    'Brown':       'قهوه‌ای',
    'Gray':        'خاکستری',
    'Black':       'مشکی',
    'White':       'سفید',
    'Beige':       'کرمی و بژ',
    'Teal':        'سبزآبی',
    'Cream':       'کرمی',
    'Peach':       'هلویی',
    'Other':       'سایر',
}


# ---------------------------------------------------------------------------
# Core conversion utilities
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_str: str) -> tuple:
    """
    Convert a hex color code to an (R, G, B) tuple of integers in [0, 255].

    Args:
        hex_str: A hex color string, e.g. '#388E3C' or '388E3C'.

    Returns:
        A tuple (R, G, B) where each value is an integer in [0, 255].

    Raises:
        ValueError: If the hex string is not exactly 6 hex digits.
    """
    hex_clean = hex_str.strip().lstrip('#')
    if len(hex_clean) != 6:
        raise ValueError(f"Invalid hex color code: '{hex_str}'. Expected 6 hex digits.")
    return tuple(int(hex_clean[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_lab(r: int, g: int, b: int) -> tuple:
    """
    Convert 8-bit sRGB (0–255) to CIE L*a*b* under the standard D65 illuminant.

    The CIE L*a*b* color space is perceptually uniform: Euclidean distance in
    this space (Delta E / ΔE) corresponds to perceived color difference.

    Conversion pipeline: sRGB → linear RGB → CIEXYZ (D65) → CIE L*a*b*

    Args:
        r, g, b: Integer channel values in [0, 255].

    Returns:
        (L*, a*, b*) as floats, rounded to 2 decimal places.
    """
    def _linearize(c_byte: int) -> float:
        """Apply sRGB gamma expansion to one channel."""
        c = c_byte / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    rl, gl, bl = _linearize(r), _linearize(g), _linearize(b)

    # sRGB → CIEXYZ (D65 white point, IEC 61966-2-1 matrix)
    x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

    # Normalize by D65 reference white
    xn, yn, zn = 0.95047, 1.00000, 1.08883

    def _f(t: float) -> float:
        """CIE standard cube-root / linear piece-wise function."""
        return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t) + (16.0 / 116.0)

    fx, fy, fz = _f(x / xn), _f(y / yn), _f(z / zn)

    L = max(0.0, 116.0 * fy - 16.0)
    a = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)

    return round(float(L), 2), round(float(a), 2), round(float(b_val), 2)


# ---------------------------------------------------------------------------
# Base color family extraction
# ---------------------------------------------------------------------------

def get_base_color(english_name: str) -> tuple:
    """
    Determine the base color family from a full English color name.

    Algorithm:
      1. Strip parenthetical remarks and lowercase the name.
      2. Look up the full cleaned name in COMPOUND_MAP.
      3. Strip true modifier words and look up the remaining words in COMPOUND_MAP.
      4. Try each regex pattern in BASE_FALLBACK_PATTERNS against the original name.
      5. Fall back to title-casing the stripped candidate.

    Args:
        english_name: Full English color name, e.g. 'Dark Navy Blue'.

    Returns:
        (Base_English, Base_Persian), e.g. ('Navy Blue', 'سرمه‌ای').
    """
    # Remove anything in parentheses and normalize to lowercase
    clean = re.sub(r'\(.*?\)', '', str(english_name)).strip().lower()

    # Step 1: exact compound-map lookup
    if clean in COMPOUND_MAP:
        return COMPOUND_MAP[clean]

    # Step 2: strip modifiers, re-check compound map
    words = [w for w in clean.split() if w not in MODIFIERS]
    candidate = ' '.join(words)
    if candidate in COMPOUND_MAP:
        return COMPOUND_MAP[candidate]

    # Step 3: regex fallback on the original (un-stripped) name
    for base_en, base_fa, pattern in BASE_FALLBACK_PATTERNS:
        if re.search(pattern, clean):
            return base_en, base_fa

    # Step 4: title-case the stripped candidate as a last resort
    if candidate:
        title_cand = ' '.join(w.capitalize() for w in candidate.split())
        return title_cand, EN_TO_FA_BASE.get(title_cand, 'سایر')

    return 'Other', 'سایر'


# ---------------------------------------------------------------------------
# Row-level transformation
# ---------------------------------------------------------------------------

def convert_color_row(row: pd.Series) -> pd.Series:
    """
    Compute all derived color-space features and the base color family for one row.

    Input columns required: 'Persian Name', 'EnglishName', 'HexCode (24 bit)'.

    Returns a pd.Series with 18 columns (see redesign_dataset docstring).
    """
    hex_code = str(row['HexCode (24 bit)']).strip()
    r, g, b = hex_to_rgb(hex_code)

    # Normalize RGB to [0.0, 1.0] for colorsys
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0

    # HSV: Hue [0, 1], Saturation [0, 1], Value [0, 1]
    h_hsv, s_hsv, v_hsv = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)

    # HLS: Hue [0, 1], Lightness [0, 1], Saturation [0, 1]
    h_hls, l_hls, s_hsl = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)

    # CIE L*a*b*
    lab_l, lab_a, lab_b = rgb_to_lab(r, g, b)

    # Base color family
    base_en, base_fa = get_base_color(row['EnglishName'])

    return pd.Series({
        'Persian Name':       row['Persian Name'],
        'EnglishName':        row['EnglishName'],
        'HexCode (24 bit)':   hex_code,
        'Red (8 bit)':        int(r),
        'Green (8 bit)':      int(g),
        'Blue (8 bit)':       int(b),
        'Hue (degrees)':      round(h_hsv * 360.0, 1),
        'HSL.S (%)':          round(s_hsl * 100.0, 1),
        'HSL.L (%)':          round(l_hls * 100.0, 1),
        'HSV.S (%)':          round(s_hsv * 100.0, 1),
        'HSV.V (%)':          round(v_hsv * 100.0, 1),
        'Lab_L':              lab_l,
        'Lab_A':              lab_a,
        'Lab_B':              lab_b,
        'Base_English':       base_en,
        'Base_Persian':       base_fa,
        # Convenience alias columns used by train.py and notebooks
        'hex':                hex_code,
        'name':               row['EnglishName'],
    })


# ---------------------------------------------------------------------------
# Dataset redesign entry point
# ---------------------------------------------------------------------------

def redesign_dataset(
    raw_csv_path: str = "data/colors-raw.csv",
    output_csv_path: str = "data/colors.csv",
) -> pd.DataFrame:
    """
    Read raw color data, compute all color-space features, and save the result.

    The function looks for the raw CSV at `raw_csv_path`; if that path doesn't
    exist it also checks the `data/` subdirectory automatically.

    Args:
        raw_csv_path:    Path to the source CSV (default: 'data/colors-raw.csv').
        output_csv_path: Path for the output enriched CSV (default: 'data/colors.csv').

    Returns:
        The transformed DataFrame.
    """
    if not os.path.exists(raw_csv_path):
        # Fallback: look in data/ subdirectory
        alt = os.path.join("data", os.path.basename(raw_csv_path))
        if os.path.exists(alt):
            raw_csv_path = alt
        else:
            raise FileNotFoundError(
                f"Source file not found: '{raw_csv_path}'. "
                "Make sure 'data/colors-raw.csv' exists."
            )

    print(f"Reading raw dataset from: {raw_csv_path} ...")
    df_raw = pd.read_csv(raw_csv_path, encoding='utf-8-sig')

    print(f"  → {len(df_raw)} rows loaded.")
    print("Computing RGB, Hue, HSL, HSV, CIE Lab, and Base Color Family ...")
    df_out = df_raw.apply(convert_color_row, axis=1)

    # Ensure output directory exists
    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_out.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved enriched dataset to: {output_csv_path}  ({len(df_out)} rows, {len(df_out.columns)} columns)")

    return df_out


if __name__ == "__main__":
    redesign_dataset(
        raw_csv_path=os.path.join(BASE_DIR, "data", "colors-raw.csv"),
        output_csv_path=os.path.join(BASE_DIR, "data", "colors.csv"),
    )
