# Contributing to KNN Color Classification

Thank you for your interest in contributing! Here's how to get started.

---

## 🛠️ Development Setup

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/kopolart/knn-color-classification.git
   cd knn-color-classification
   ```
3. Create a **virtual environment** and install dependencies:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   pip install -r requirements.txt
   ```
4. Create a **feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## 📋 Ways to Contribute

### 🐛 Bug Reports
- Open a [GitHub Issue](https://github.com/kopolart/knn-color-classification/issues) with a clear title and description.
- Include steps to reproduce, expected behavior, and actual behavior.
- Attach relevant error messages or stack traces.

### 💡 Feature Suggestions
- Open an Issue first to discuss the idea before implementing.
- Good candidates: new color spaces, larger datasets, UI demos, REST API.

### 📦 Dataset Contributions
- If you want to add more colors to `data/colors-raw.csv`, make sure each row includes:
  - `Persian Name` — the color name in Persian (Farsi)
  - `EnglishName` — the color name in English
  - `HexCode (24 bit)` — a valid 6-digit hex color (e.g. `#3A7CA5`)
- Run `python prepare_dataset.py` to regenerate `colors.csv` from the updated raw file.
- Re-run `python train.py` to retrain the model with the new data.
- Dataset contributions must preserve the original attribution and CC BY 4.0 terms:
  **کپل‌آرت – Ali Yousefi** with a link to <https://kopolart.ir/colors>.

### 🔬 Code Improvements
- Please follow existing code style and conventions.
- Add docstrings to any new functions you introduce.
- Run `python train.py` and verify the model still trains correctly after your changes.

---

## 🔄 Pull Request Process

1. Make sure your branch is up to date with `main`:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
2. Commit your changes with a clear, descriptive commit message:
   ```bash
   git commit -m "feat: add support for CMYK color space features"
   ```
3. Push your branch and open a Pull Request against `main`.
4. In the PR description, explain:
   - **What** you changed and **why**.
   - How to test the change.
   - Any known limitations.

---

## 📝 Commit Message Convention

Use the following prefixes:

| Prefix | When to use |
|--------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `refactor:` | Code refactor without behavior change |
| `data:` | Dataset additions or modifications |
| `chore:` | Tooling, dependencies, CI |

---

## 🙏 Code of Conduct

Be respectful and constructive. This project welcomes contributions from everyone regardless of background or experience level.
