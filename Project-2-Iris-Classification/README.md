# 🌸 Iris Data Classification Using AI
### Project 2 — DecodeLabs Industrial Training Kit | Batch 2026

---

## 📌 Project Overview

This project demonstrates a complete **supervised machine learning pipeline** using the classic Iris flower dataset. The goal is to train a K-Nearest Neighbors (KNN) classifier that can identify the species of an Iris flower based on its physical measurements.

---

## 🎯 Objectives

- Load and explore a real-world dataset
- Apply feature scaling to prepare data for distance-based algorithms
- Split data into training and testing sets
- Train a KNN classification model
- Evaluate performance using Confusion Matrix and F1 Score
- Predict the species of brand-new flower samples

---

## 🗂️ Project Structure

```
iris-classification/
│
├── iris_classification.ipynb   # Main Colab notebook (full pipeline)
└── README.md                   # Project documentation (this file)
```

---

## 🌿 Dataset — The Iris Benchmark

| Property | Value |
|----------|-------|
| Source | `sklearn.datasets.load_iris()` |
| Total Samples | 150 (balanced) |
| Classes | 3 |
| Features | 4 |
| Class Names | Setosa, Versicolor, Virginica |

**Features (measurements in cm):**
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

## ⚙️ Pipeline — IPO Framework

```
INPUT  →  PROCESS  →  OUTPUT
```

| Stage | Steps |
|-------|-------|
| **Input** | Load Iris dataset → Feature Scaling (StandardScaler) |
| **Process** | Train/Test Split (80/20) → KNN Algorithm |
| **Output** | Confusion Matrix → F1 Score |

---

## 🔬 Algorithm — K-Nearest Neighbors (KNN)

> *"Similar things exist in close proximity."*

KNN classifies a new data point by looking at its **K closest neighbors** in the training set and assigning the majority class among them.

- **K = 5** was selected as the optimal value using the **Elbow Method**
- Distance metric: Euclidean distance
- Scaling is **required** before applying KNN to remove feature bias

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | **100%** |
| F1 Score (weighted) | **1.0000** |

### K Comparison Table

| K | Accuracy | F1 Score |
|---|----------|----------|
| 1 | 96.67% | 0.9664 |
| 3 | 100.00% | 1.0000 |
| 5 | 100.00% | 1.0000 |
| 7 | 100.00% | 1.0000 |
| 9 | 100.00% | 1.0000 |
| 11 | 100.00% | 1.0000 |

### Confusion Matrix

```
              setosa  versicolor  virginica
    setosa      10          0          0
versicolor       0          9          0
 virginica       0          0         11
```

Zero misclassifications on the test set.

---

## 🚀 How to Run

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `iris_classification.ipynb`
3. Select **Runtime → Run All**
4. No additional installations needed — all libraries are pre-installed on Colab

---

## 📦 Libraries Used

| Library | Purpose |
|---------|---------|
| `scikit-learn` | Dataset, scaling, model, metrics |
| `numpy` | Numerical operations |
| `pandas` | Data exploration via DataFrame |
| `matplotlib` | Elbow curve visualization |
| `seaborn` | Confusion matrix heatmap |

---

## 🧠 Key Concepts Covered

- **Supervised Learning** — learning from labeled data
- **Feature Scaling** — StandardScaler (mean=0, variance=1)
- **Train/Test Split** — 80% train / 20% test with shuffle
- **Overfitting vs Underfitting** — demonstrated via Elbow Method
- **Confusion Matrix** — TP, FP, FN, TN breakdown
- **F1 Score** — harmonic mean of precision and recall

---
