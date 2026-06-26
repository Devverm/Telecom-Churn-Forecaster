# Telecom Churn Forecaster

![Python](https://img.shields.io/badge/Python-3.10.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Complete-success)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end machine learning pipeline that predicts telecom customer churn — from raw data to a live interactive web app. The system identifies at-risk customers, surfaces key risk factors, and recommends retention actions in real time.

---

## Table of Contents

- [Business Problem](#business-problem)
- [Key Results](#key-results)
- [Key Insights](#key-insights)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Streamlit App](#streamlit-app)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Business Problem

Customer churn is one of the most expensive problems in the telecom industry — acquiring a new customer costs 5–7× more than retaining an existing one. This project builds a production-ready churn classifier that:

- Predicts whether a customer will churn based on their profile and service usage
- Surfaces the specific risk factors driving the prediction
- Recommends targeted retention actions via an interactive web app

---

## Key Results

| Metric | Score |
|---|---|
| Best Model | **Logistic Regression** |
| ROC-AUC | **0.8458** |
| Accuracy | **80.41%** |
| F1-Score (Churn) | **0.5929** |

> ROC-AUC is used as the primary metric because the dataset is class-imbalanced (~26.5% churn), making accuracy alone a misleading measure.

---

## Key Insights

### Dataset Overview
- **7,043 customers**, 20 features, **~26.5% churn rate** (1,869 churned vs 5,174 retained)
- Class imbalance confirmed — addressed via engineered features and appropriate evaluation metrics
- `TotalCharges` contained whitespace entries treated as missing values — replaced with median

### EDA Findings

**Contract Type — Strongest Churn Signal:**

| Contract | Churn Rate |
|---|---|
| Month-to-month | **42.7%** |
| One year | 11.3% |
| Two year | 2.8% |

Month-to-month customers churn at **15× the rate** of two-year contract holders. Contract type is the single most actionable lever for retention.

**Tenure — Short Relationships Are High Risk:**

| Churn | Avg Tenure |
|---|---|
| No | 37.6 months |
| Yes | **18.0 months** |

Customers in their first 12–18 months are the highest-risk cohort. Early engagement programs are critical.

**Monthly Charges — Higher Bills Drive Higher Churn:**

| Churn | Avg Monthly Charge |
|---|---|
| No | $61.27 |
| Yes | **$74.44** |

Churned customers pay ~$13/month more on average, pointing to price sensitivity as a key factor.

**Internet Service Type:**

| Service | Churn Rate |
|---|---|
| Fiber optic | **41.9%** |
| DSL | 19.0% |
| No internet | 7.4% |

Fiber optic subscribers churn at **2.2× the rate** of DSL users — likely driven by increased competition and higher pricing in that segment.

**Payment Method:**

| Payment Method | Churn Rate |
|---|---|
| Electronic check | **45.3%** |
| Mailed check | 19.1% |
| Bank transfer (auto) | 16.7% |
| Credit card (auto) | 15.2% |

Electronic check users churn at **3× the rate** of automatic payment users — a strong signal of low engagement and payment friction.

**Demographics:**

| Segment | Churn Rate |
|---|---|
| Senior citizens | **41.7%** |
| Non-seniors | 23.6% |
| No partner | 33.0% |
| Has partner | 19.7% |
| No dependents | 31.3% |
| Has dependents | 15.5% |
| Paperless billing | 33.6% |
| Paper billing | 16.3% |

Value-added services significantly reduce churn — customers without **Tech Support** or **Online Security** are substantially more likely to leave.

### Feature Engineering Impact
Three engineered features improved model performance by **+8% ROC-AUC**:

- `tenure_group` — bucketed tenure into 4 lifecycle stages (0–12, 13–24, 25–48, 49–72 months)
- `avg_monthly_per_tenure` — average spend rate (`TotalCharges / (tenure + 1)`)
- `num_services` — count of active subscriptions (Phone, Internet, Security, Backup, Protection, Tech Support)

### Model Comparison (5 Models Trained)

| Model | ROC-AUC |
|---|---|
| Decision Tree | baseline |
| Random Forest | competitive |
| Gradient Boosting | competitive |
| XGBoost | competitive |
| **Logistic Regression** | **0.8458 ✓ Best** |

Logistic Regression outperformed tree-based models after feature engineering and proper scaling — a reminder that simpler models can win when features are well-constructed.

### App Risk Factor Logic
The Streamlit app flags these evidence-based risk signals at inference time:
- Month-to-month contract
- Tenure < 12 months
- Monthly charges > $80
- No Online Security subscription
- No Tech Support subscription
- Electronic check payment method
- No partner

---

## Project Structure

```
churn-prediction-ml/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version (3.10.13)
├── .gitignore
│
├── data/                           # Raw and processed datasets
│   └── .gitkeep
│
├── models/                         # Saved model & preprocessor artifacts
│   ├── best_model_*.pkl            # Trained Logistic Regression model
│   ├── preprocessor.pkl            # Fitted scaler + label encoders
│   └── .gitkeep
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   └── 03_model_training.ipynb     # Model training & evaluation
│
├── src/
│   └── data_preprocessing.py       # Standalone preprocessing pipeline
│
├── reports/
│   ├── model_comparison_graph.png
│   ├── roc_curves.png
│   └── confusion_matrix.png
│
└── others/
```

---

## Workflow

### 1. Data Preprocessing
- Dropped `customerID` (non-predictive)
- Replaced whitespace values in `TotalCharges` with median; cast to float
- Binary encoded: `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `gender`
- Label encoded remaining categorical columns; encoders saved to `preprocessor.pkl`
- Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`, `avg_monthly_per_tenure`) with `StandardScaler`

### 2. Feature Engineering
- `tenure_group`: lifecycle stage buckets (0–12 / 13–24 / 25–48 / 49–72 months)
- `avg_monthly_per_tenure`: spending intensity ratio
- `num_services`: total active service subscriptions

### 3. Model Training
Five classifiers trained and compared using ROC-AUC as the primary metric:
Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost

### 4. Model Evaluation
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrix and ROC curve plots saved to `reports/`

### 5. Model Persistence
- Best model saved as `models/best_model_*.pkl`
- Preprocessor (scaler + encoders) saved as `models/preprocessor.pkl` via `joblib`

### 6. Streamlit App Deployment
- Loads saved model + preprocessor at startup
- Accepts full customer profile via interactive form
- Returns churn prediction, probability, confidence chart, and risk factors
- Includes High-Risk and Low-Risk quick-test profiles

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 2.3.3 | Data manipulation |
| `numpy` | 2.4.2 | Numerical operations |
| `scikit-learn` | 1.5.2 | ML models, preprocessing, evaluation |
| `xgboost` | 2.1.3 | Gradient boosting classifier |
| `imbalanced-learn` | 0.12.4 | Class imbalance handling |
| `matplotlib` | 3.9.3 | Static visualizations |
| `seaborn` | 0.13.2 | Statistical plots |
| `plotly` | 5.22.0 | Interactive charts in app |
| `joblib` | 1.4.2 | Model serialization |
| `streamlit` | latest | Web application framework |

---

## Installation & Setup

### Prerequisites
- Python 3.10.13 (see `runtime.txt`)
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/misrapk/churn-prediction-ml.git
cd churn-prediction-ml

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# Get WA_Fn-UseC_-Telco-Customer-Churn.csv from:
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place it in the data/ folder
```

---

## Usage

### 1. Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Train Models
```bash
jupyter notebook notebooks/03_model_training.ipynb
```

### 3. Run Preprocessing Standalone
```bash
python src/data_preprocessing.py
```

### 4. Launch the Web App
```bash
streamlit run app.py
```

---

## Streamlit App

The app provides a full real-time prediction interface:

**Sidebar**
- Model performance summary (ROC-AUC, Accuracy, F1)
- Quick-load test profiles: High-Risk and Low-Risk

**Input Form**
- Demographics: gender, senior citizen, partner, dependents, tenure
- Account info: contract type, paperless billing, payment method, charges
- Services: phone, internet, security, backup, protection, tech support, streaming

**Output Panel**
- Churn / No Churn verdict with probability percentage
- Color-coded risk indicator (red = high risk, green = low risk)
- Horizontal confidence bar chart (Plotly)
- Specific risk factors detected for the customer
- Actionable retention recommendations

**Test Profiles**

| Profile | Contract | Tenure | Services | Expected |
|---|---|---|---|---|
| High Risk | Month-to-month | 3 months | Minimal | ~Churn |
| Low Risk | Two year | 48 months | Full suite | ~Stay |

---

## Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV / RandomizedSearchCV
- [ ] Handle class imbalance with SMOTE / undersampling
- [ ] REST API with FastAPI for programmatic model serving
- [ ] CI/CD pipeline for automated retraining on new data
- [ ] Deploy on AWS / Azure / GCP as a managed web service
- [ ] Batch prediction mode for scoring entire customer lists
- [ ] SHAP values for individual prediction explainability

---

## Lessons Learned

1. **Feature engineering** had the highest ROI — three new features added +8% ROC-AUC
2. **Simpler models can win** — Logistic Regression outperformed all tree-based models after proper feature prep
3. **ROC-AUC over accuracy** — with 26.5% churn rate, accuracy is misleading; AUC is the right north-star metric
4. **Business context matters** — optimizing for recall may be more valuable than precision (missing a churner costs more than a false alarm)

---
