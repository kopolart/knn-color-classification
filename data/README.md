# Dataset — KopolArt Color Collection

This directory contains the color data used by the two-stage KNN classifier.

## Source and attribution

- **Source:** [KopolArt Color Collection](https://kopolart.ir/colors)
- **Curator / author:** Ali Yousefi (KopolArt)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Required attribution:** **کپل‌آرت – Ali Yousefi** with a direct link to <https://kopolart.ir/colors>

If you reuse, redistribute, or build on this dataset, keep the attribution above
and link back to <https://kopolart.ir/colors>. The CC BY 4.0 license permits
commercial and non-commercial use, adaptation, and redistribution with
attribution.

The **code** in this repository is licensed separately under the MIT License
(see the repository-root `LICENSE`). The dataset is **not** covered by the MIT
license; it remains under CC BY 4.0.

## Files

| File | Rows | Columns | Description |
|---|---:|---:|---|
| `colors-raw.csv` | 608 | 3 | Raw source: Persian name, English name, hex code |
| `colors.csv` | 608 | 18 | Enriched: RGB, Hue, HSL, HSV, CIE Lab, base family, aliases |

The beginner-friendly notebook and demo in `simple_version/` resolve the dataset
from this directory (`../data/colors.csv`), so there is a single source of truth.

## Schema of `colors.csv`

| Column | Description |
|---|---|
| `Persian Name` | Color name in Persian (Farsi) |
| `EnglishName` | Color name in English |
| `HexCode (24 bit)` | Hex color code, e.g. `#388E3C` |
| `Red (8 bit)` / `Green (8 bit)` / `Blue (8 bit)` | RGB channels in `[0, 255]` |
| `Hue (degrees)` | Hue angle in `[0, 360)` |
| `HSL.S (%)` / `HSL.L (%)` | HSL saturation and lightness |
| `HSV.S (%)` / `HSV.V (%)` | HSV saturation and value |
| `Lab_L` / `Lab_A` / `Lab_B` | CIE L\*a\*b\* coordinates |
| `Base_English` | Coarse color family in English |
| `Base_Persian` | Coarse color family in Persian |
| `hex` / `name` | Convenience aliases used by the training scripts |

## Regenerating the enriched file

```bash
python prepare_dataset.py
```

This reads `data/colors-raw.csv` and writes `data/colors.csv`.

## Citation

```bibtex
@misc{kopolart_colors_2026,
  author = {Yousefi, Ali},
  title  = {KopolArt Color Collection},
  year   = {2026},
  url    = {https://kopolart.ir/colors},
  note   = {Licensed under CC BY 4.0; attribution: کپل‌آرت – Ali Yousefi}
}
```
