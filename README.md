# AI-Driven Security Analysis of Encrypted Network Traffic (Research Apprenticeship @ UTP Malaysia)

This repository contains the complete codebase and reproducible pipeline developed during my research apprenticeship at **Universiti Teknologi PETRONAS (UTP), Malaysia**, which led to the conference paper *"AI-Driven Security Analysis of Encrypted Communications"*. The project focuses on analyzing **encrypted network traffic metadata** (flow-level features) for anomaly detection using machine learning — **without decrypting payloads**, thus preserving user privacy.

---

## 📌 Overview

Traditional intrusion detection systems struggle with encrypted traffic, as payload inspection is impossible. This project leverages **flow metadata** (packet counts, sizes, timings) from the [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html) to classify benign vs. malicious activity.

**Key contributions implemented in this repo:**

* 🛠️ **End-to-end ML pipeline** — train-only preprocessing (column drops, global mean imputation, incremental scaling), streaming data loaders, and baseline training scripts.
* 🌲 **Baseline models** — Decision Tree, Linear Support Vector Machine (LinearSVC), and Stochastic Gradient Descent (SGD with threshold calibration).
* 📊 **Comprehensive evaluation** — Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC, and confusion matrices.
* 🔎 **Feature interpretability suite** — per-feature Cohen’s d, point-biserial correlation, and model-derived importances, exported to CSV and Markdown for reporting.

These results directly support the claims in the conference paper.

---

## 🗂️ Repository Structure

```
.
├── src/                     # Core pipeline code
│   ├── data_loader.py       # Stream CSVs in memory-safe chunks
│   ├── preprocess.py        # Global column drops + imputation
│   ├── scaling.py           # Incremental StandardScaler fitting
│   ├── run_full_preprocessing.py # Train-only preprocessing & artifact saving
│   ├── train_test_model.py  # SGDClassifier (with threshold tuning)
│   ├── baselines_full_dataset.py # Train DT & LinearSVC on full dataset
│   ├── baselines_compare_sampled.py # Baselines on stratified sample
│   ├── objective1_analysis.py # Feature interpretability (Cohen's d, r_pb, weights)
│   └── export_svm_cm.py     # Export LinearSVC confusion matrix as PNG
│
├── scripts/
│   └── make_sample.py       # Helper: create demo CSV (10 rows × 79 cols)
│
├── data/
│   └── CICIDS2017_sample.csv # Small demo sample (safe to push)
│
├── reports/                 # Generated outputs (top-15 features CSV, MD summaries)
├── requirements.txt         # Python dependencies
├── README.md                # (this file)
└── .gitignore               # Excludes full dataset & large artifacts
```

---

## 🚀 Quickstart Demo

This demo runs on the included **tiny sample dataset** (`data/CICIDS2017_sample.csv`) to showcase the pipeline in under a minute.

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate demo sample (already included, but reproducible)
python scripts/make_sample.py

# 3. Run preprocessing (train-only split, scaler, artifacts)
python src/run_full_preprocessing.py

# 4. Train baseline SGD classifier and evaluate
python src/train_test_model.py

# 5. Run feature analysis (Cohen’s d, r_pb, model weights)
python src/objective1_analysis.py
```

Outputs (CSV + Markdown) will appear under `reports/`, and figures under `figures/`.

---

## 📈 Full Dataset Reproduction

The full **CICIDS2017 dataset** is not hosted here due to size. To run full experiments:

1. Download from the [official CICIDS2017 site](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Place CSVs under `data/CICIDS2017_CSVs/`.
3. Rerun the steps above (preprocessing will automatically detect all files).

Artifacts (trained models, full metrics) are excluded from the repo (`.gitignore`). You can regenerate them locally.

---

## 📊 Example Results (from Conference Paper)

Baseline test performance (full dataset):

| Model          | Accuracy | Precision | Recall | F1     | PR-AUC | ROC-AUC |
| -------------- | -------- | --------- | ------ | ------ | ------ | ------- |
| DecisionTree   | 0.6996   | 0.9620    | 0.2799 | 0.4336 | 0.5651 | 0.6361  |
| LinearSVC      | 0.7552   | 0.8918    | 0.4600 | 0.6069 | 0.6740 | 0.5882  |
| SGD (thr=0.05) | 0.7445   | 0.8704    | 0.4441 | 0.5882 | 0.6762 | 0.6875  |

Feature analysis highlighted **Flow Duration**, **Total Forward/Backward Packets**, and **Inter-arrival times** as top discriminators.

---

## 📚 Reference

If you use this code, please cite the paper:

**Lakshmi Narasimham Rallabandi, Dr. Fasee Ullah, Shashi Bhushan.**
*AI-Driven Security Analysis of Encrypted Communications*.
Presented at \[Conference Name], 2025.

---

## 🔒 Ethics & Privacy

* **No decryption performed** — analysis uses only metadata.
* Aligns with privacy-by-design: preserves user confidentiality while detecting anomalies.

---

## 📜 License
MIT License — free to use with attribution.  
See [LICENSE](./LICENSE) for full details.

---

## 🙌 Acknowledgments

* **Universiti Teknologi PETRONAS (UTP)** – Research Apprenticeship support
* **SRM University, AP** – Undergraduate program support
* **Canadian Institute for Cybersecurity** – For releasing CICIDS2017 dataset