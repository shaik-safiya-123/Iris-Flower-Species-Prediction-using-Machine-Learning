# 🌸 Iris Flower Species Classifier

A **Final Year Machine Learning Project** that classifies Iris flower species using 6 different ML algorithms and deploys the best model via an interactive Streamlit web application with a premium glassmorphic UI.

---

## 🚀 Live Demo

Run locally with:
```bash
python train.py
streamlit run project.py
```

---

## 🤖 ML Models Trained

| Model | Test Accuracy | Fit Status |
|---|---|---|
| ⭐ KNN (k=3) | 100.0% | ✅ Best Model |
| SVM (SVC) | 96.7% | ✅ Well-fitted |
| Logistic Regression | 96.7% | ✅ Well-fitted |
| Gaussian Naive Bayes | 96.7% | ✅ Well-fitted |
| Decision Tree | 93.3% | ⚠️ Overfitting |
| Random Forest | 90.0% | ⚠️ Overfitting |

**Train/Test Split:** 80% training / 20% testing (stratified)

---

## 🎨 UI Features

- 🌑 Dark glassmorphic UI with Google Fonts (Poppins)
- 🎚️ Interactive sliders for all 4 flower measurements
- 📊 Real-time confidence probability bars per class
- 🔄 Live model switcher across all 6 trained classifiers
- 🌸 Flower image + botanical description on prediction

---

## 🛠️ Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python train.py

# Launch app
streamlit run project.py
```

---

## 📦 Tech Stack

`Python` · `Scikit-learn` · `Streamlit` · `NumPy` · `Pandas` · `HTML/CSS`

---

## 📖 Full Guide

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for complete documentation.
