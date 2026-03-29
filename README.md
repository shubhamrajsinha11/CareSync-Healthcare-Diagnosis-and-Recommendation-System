# ⚕️ CareSync — Healthcare Diagnosis & Recommendation System

> A comprehensive AI-powered healthcare web application that predicts diseases, recommends medicines, generates personalised nutrition plans, and creates daily routines — all in one place.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![SQLite](https://img.shields.io/badge/SQLite-Built--in-green?style=flat-square&logo=sqlite)
![License](https://img.shields.io/badge/License-Educational-lightgrey?style=flat-square)

---

## 📌 About the Project

CareSync is a B.Tech Major Project developed at **Sardar Beant Singh State University, Gurdaspur**. It bridges the gap between individuals and timely healthcare guidance by providing an accessible, data-driven platform for disease prediction and personalised health recommendations.

Instead of relying on generic internet searches, CareSync uses a trained **machine learning ensemble** to analyse symptoms and return structured, reliable health insights — instantly.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔒 **User Authentication** | Secure register/login with PBKDF2-SHA256 password hashing and SQLite storage |
| 🔬 **Symptom Analysis** | Predicts top 3 diseases from 130+ symptoms across 40+ categories with confidence scores |
| 📊 **Risk Scoring** | Multi-factor risk assessment using symptoms, BMI, age, and body temperature |
| 💊 **Medication Recommendations** | OTC, prescription, natural remedies, dosage, contraindications, and treatment lines |
| 🥗 **7-Day Nutrition Plan** | Personalised meal plans with foods to eat/avoid, recipes, and exercise suggestions |
| 🗓️ **Daily Routine Generator** | Hour-by-hour schedule with medication timing, hydration, exercise, and rest periods |
| 📈 **Health Analytics** | Interactive Plotly charts for BMI, temperature, symptom frequency, and risk profile |
| 📋 **Clinical Report** | Downloadable patient summary with diagnosis, risk level, and recommended actions |
| 🕑 **Prediction History** | All past predictions saved to SQLite — accessible across sessions |
| 👤 **Patient Profile** | Persistent health profile that pre-fills analysis forms automatically |

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit + Custom CSS (Dark Theme)
- **Backend:** Python 3.8+
- **ML Models:** Scikit-learn (Random Forest, Logistic Regression, Decision Tree)
- **Data Processing:** Pandas, NumPy
- **Visualisation:** Plotly
- **Database:** SQLite (built-in, no setup required)
- **Security:** PBKDF2-SHA256 password hashing

---

## 📁 Project Structure

```
CareSync/
│
├── app.py                    ← Main Streamlit app (all 9 pages)
├── auth.py                   ← Login, register, session, sidebar widget
├── database.py               ← All SQLite operations
├── history.py                ← Prediction history page
├── profile.py                ← Patient profile settings page
├── train_models.py           ← One-time ML model training script
├── requirements.txt          ← Python dependencies
├── README.md
│
├── utils/
│   ├── prediction.py         ← ML prediction & risk scoring
│   ├── preprocessing.py      ← Dataset loading & feature engineering
│   ├── medicine_recommender.py
│   ├── diet_planner.py
│   ├── routine_generator.py
│   └── analytics.py          ← Plotly health charts
│
├── data/                     ← CSV datasets (not included in repo)
└── models/                   ← Trained model artifacts (not included in repo)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/shubhamrajsinha11/caresync.git
cd caresync
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Add datasets
Place the following CSV files inside the `data/` folder:
```
data/
├── Testing.csv
├── Final_Augmented_dataset_Diseases_and_Symptoms.csv
├── Symptom-severity.csv
├── medical data.csv
└── drugsComTest_raw.csv
```

### Step 4 — Train the model (run only once)
```bash
python train_models.py
```
This will train Random Forest, Logistic Regression, and Decision Tree classifiers, select the best one, and save the model to `models/`. This step only needs to be run once.

### Step 5 — Launch the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 ML Model Details

| Model | Configuration | Accuracy |
|---|---|---|
| Random Forest | 100 trees, max_depth=20 | 88–95% |
| Logistic Regression | max_iter=1000, C=1.0 | 82–90% |
| Decision Tree | max_depth=20, max_features=sqrt | 80–88% |

The training pipeline applies **mutual information feature selection** (496 → 200 features), **StandardScaler**, and **PCA** (95% variance retained) before training all three models. The best-performing model is automatically selected and saved.

---

## 🗄️ Database Schema

CareSync uses SQLite with 4 tables:

- **users** — Account credentials and metadata
- **patient_profiles** — Health details (age, weight, height, blood group)
- **prediction_history** — All past predictions with symptoms stored as JSON
- **health_metrics** — BMI, temperature, and risk score snapshots per session

---

## ⚠️ Disclaimer

CareSync is developed for **educational and informational purposes only**. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.

---

## 👨‍💻 Author

**Shubham Raj Sinha**
URN: 22303271
Department of Computer Science & Engineering
Sardar Beant Singh State University, Gurdaspur

---

## 📝 License

This project is provided as-is for educational purposes.
