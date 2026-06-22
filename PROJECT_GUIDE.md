# 🌸 Iris Flower Species Classifier — Complete Project Guide

---

## 📁 Project Overview

This is a **Final Year Machine Learning Project** that classifies Iris flower species
(Setosa, Versicolor, Virginica) using 6 different ML algorithms and deploys the best
model through an interactive Streamlit web application with a premium glassmorphic UI.

---

## 📂 Final Project Structure

```
Final-Year-Project-main/
│
├── Iris Flowers/                          ← Flower images used in the UI
│   ├── Irissetosa1.jpg
│   ├── Iris_versicolor__flo_npyvSQOSVQ8O.jpeg
│   └── iris_virginica_virginica_lg.jpg
│
├── Iris Project.ipynb                     ← Original exploratory notebook (reference)
├── train.py                               ← ✅ NEW: Trains all 6 ML models
├── project.py                             ← ✅ NEW: Streamlit web dashboard
├── models.pkl                             ← ✅ NEW: All 6 trained models + metrics
├── Data.pkl                               ← ✅ NEW: Best model (KNN k=3)
├── requirements.txt                       ← Python dependencies
└── PROJECT_GUIDE.md                       ← This document
```

---

## 🧹 What Was Cleaned Up

| File | Action | Reason |
|---|---|---|
| `Rough.ipynb` | ❌ Deleted | Rough draft, not needed |
| `Team 7 Logbook.pdf` | ❌ Deleted | Removed as requested |
| `Iris Project.ipynb` | ✅ Kept | Original reference notebook |

---

## 🤖 Machine Learning Pipeline

### Dataset
- **Source:** Scikit-learn built-in `load_iris()` dataset
- **Features:** 4 numerical features per sample
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Target Classes:** 3 species
  - `0` → Iris Setosa
  - `1` → Iris Versicolor
  - `2` → Iris Virginica
- **Total Samples:** 150 (50 per class)

### Train / Test Split
```
Training Data  : 80%  → 120 samples
Testing Data   : 20%  → 30 samples
Strategy       : Stratified split (equal class proportions in both sets)
Random State   : 42 (for reproducibility)
```

---

## 📊 Model Training Results

All 6 algorithms were trained and evaluated. Results on the **80/20 split**:

| # | Model | Train Acc | Test Acc | Train Err | Test Err | Fit Status |
|---|---|---|---|---|---|---|
| 1 | ⭐ **KNN (k=3)** | 95.8% | **100.0%** | 4.2% | **0.0%** | ✅ Well-fitted |
| 2 | SVM (SVC) | 98.3% | 96.7% | 1.7% | 3.3% | ✅ Well-fitted |
| 3 | Logistic Regression | 97.5% | 96.7% | 2.5% | 3.3% | ✅ Well-fitted |
| 4 | Gaussian Naive Bayes | 95.8% | 96.7% | 4.2% | 3.3% | ✅ Well-fitted |
| 5 | Decision Tree | 100.0% | 93.3% | 0.0% | 6.7% | ⚠️ Overfitting |
| 6 | Random Forest | 100.0% | 90.0% | 0.0% | 10.0% | ⚠️ Overfitting |

### 🏆 Best Model: KNN (k=3) — 100% Test Accuracy

---

## 🔬 Overfitting & Underfitting Analysis

### Overfitting (Decision Tree & Random Forest)
- Both scored **100% training accuracy** but dropped significantly on test data.
- This means the models **memorised** the training examples rather than learning
  general patterns — a classic sign of overfitting on small datasets.
- **Fix options (not applied here):** Pruning the tree, limiting `max_depth`,
  increasing `min_samples_split`, or adding cross-validation.

### Underfitting
- **None observed** in this project. All models achieved above 90% on test data.

### Well-fitted Models
- **KNN (k=3), SVM, Logistic Regression, Gaussian Naive Bayes** all showed a
  healthy balance between training and testing accuracy — neither memorising
  nor under-learning the dataset.

---

## 🛠️ Step-by-Step Setup & Run Guide

### Step 1 — Prerequisites
Make sure Python is installed. This project uses **Python 3.13**.
Check your version:
```powershell
python --version
```

### Step 2 — Install Dependencies
Open a terminal in the project folder and run:
```powershell
pip install -r requirements.txt
```
This installs: `numpy`, `pandas`, `scikit-learn`, `streamlit`, `seaborn`, `matplotlib`

### Step 3 — Train the Models
Run the training script once to generate `models.pkl` and `Data.pkl`:
```powershell
C:\Users\shaik\AppData\Local\Programs\Python\Python313\python.exe train.py
```

**Expected output:**
```
--- Training and Evaluating ML Models (80/20 Split) ---
Model                     | Train Acc | Test Acc | Train Err | Test Err | Fitting Status
------------------------------------------------------------------------------------------
Logistic Regression       | 0.9750    | 0.9667   | 0.0250    | 0.0333   | Well-fitted (Good)
Decision Tree             | 1.0000    | 0.9333   | 0.0000    | 0.0667   | Slightly Overfitting
KNN (k=3)                 | 0.9583    | 1.0000   | 0.0417    | 0.0000   | Well-fitted (Good)
SVM (SVC)                 | 0.9833    | 0.9667   | 0.0167    | 0.0333   | Well-fitted (Good)
Random Forest             | 1.0000    | 0.9000   | 0.0000    | 0.1000   | Slightly Overfitting
Gaussian Naive Bayes      | 0.9583    | 0.9667   | 0.0417    | 0.0333   | Well-fitted (Good)

Best Model: KNN (k=3) (Test Accuracy: 1.0000)
Saved all models and metrics to 'models.pkl'
Saved the best model (KNN (k=3)) to 'Data.pkl'
```

### Step 4 — Launch the Web Application
```powershell
C:\Users\shaik\AppData\Local\Programs\Python\Python313\python.exe -m streamlit run project.py
```

Then open your browser and go to:
```
http://localhost:8501
```

---

## 🎨 UI Features & How to Use

### Sidebar (Left Panel)
| Feature | Description |
|---|---|
| **Model Selector** | Dropdown to switch between all 6 trained ML classifiers |
| **Test Acc** | Accuracy of the selected model on unseen test data |
| **Test Error** | Error rate = 1 − Test Accuracy |
| **Fit Status** | Auto-diagnoses overfitting or well-fitted state |
| **Algorithm Table** | Full comparison of all 6 models at a glance |

### Main Panel
| Feature | Description |
|---|---|
| **Sepal Length slider** | Range: 4.0 – 8.0 cm (default: 5.8) |
| **Sepal Width slider** | Range: 2.0 – 4.5 cm (default: 3.0) |
| **Petal Length slider** | Range: 1.0 – 7.0 cm (default: 3.8) |
| **Petal Width slider** | Range: 0.1 – 2.5 cm (default: 1.2) |
| **CLASSIFY FLOWER** | Runs the selected ML model and shows the result |

### Prediction Output Card
After clicking **CLASSIFY FLOWER**:
1. The **species name** is displayed in large text
2. A **botanical description** of the predicted species is shown
3. **Probability bars** show the classifier's confidence for all 3 classes
4. The **flower image** is displayed below the card

---

## 🌸 Iris Species Reference Guide

### Iris Setosa (Class 0)
- **Habitat:** Arctic and subarctic zones (Alaska, Canada, Russia)
- **Key Feature:** Very small, scale-like petals — almost invisible
- **Sepal Size:** 4.3 – 5.8 cm length
- **Petal Size:** 1.0 – 1.9 cm length
- **Distinguishing Trait:** Easily separable from the other two species

### Iris Versicolor (Class 1)
- **Habitat:** Eastern North America — wet meadows and marshes
- **Key Feature:** Blue-violet petals with yellow and white veining
- **Sepal Size:** 4.9 – 7.0 cm length
- **Petal Size:** 3.0 – 5.1 cm length
- **Common Name:** Harlequin Blue Flag

### Iris Virginica (Class 2)
- **Habitat:** Coastal wetlands and marshes of the eastern USA
- **Key Feature:** Largest petals and tallest stem of the three species
- **Sepal Size:** 5.9 – 7.9 cm length
- **Petal Size:** 4.5 – 6.9 cm length
- **Common Name:** Virginia Iris

---

## 📦 requirements.txt Contents

```
numpy
pandas
streamlit
seaborn
matplotlib
scikit-learn
```

Install with:
```powershell
pip install -r requirements.txt
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` using Python 3.13 |
| `models.pkl not found` | Run `train.py` first before launching `project.py` |
| App not opening | Make sure port 8501 is free; try `streamlit run project.py --server.port 8502` |
| Blank prediction output | Check that `Iris Flowers/` folder exists with all 3 images |
| Streamlit version issues | Run `pip install streamlit --upgrade` |

---

## 👨‍💻 Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.13 | Core programming language |
| Scikit-learn | ML model training & evaluation |
| Streamlit | Web application framework |
| NumPy / Pandas | Data handling |
| Pickle | Model serialisation |
| HTML + CSS | UI styling injected via `st.markdown` |
| Google Fonts (Poppins) | Premium typography |

---

*Document generated as part of the Final Year Project — Iris Species Classifier*
