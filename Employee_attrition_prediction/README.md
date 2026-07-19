# Employee Attrition Prediction

Predicting which employees are likely to leave a company using the IBM HR Analytics Employee Attrition dataset, with a focus on statistically validating findings rather than relying on visual inspection alone.

## Dataset
[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — 1,470 employee records, 35 features, target: `Attrition` (Yes/No).

## Project Structure
```
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   └── HR-Employee-Attrition_cleaned.csv
├── notebooks/
│   ├── 1_data_cleaning.ipynb
│   ├── 2_eda.ipynb
│   └── 3_model.ipynb
├── requirements.txt
└── README.md
```

## Pipeline

**1. Data Cleaning** — Verified no missing values or duplicates, ran logical consistency checks (e.g. `YearsInCurrentRole` ≤ `YearsAtCompany`), dropped constant/non-informative columns (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`), and flagged the ~84/16 class imbalance in the target.

**2. Exploratory Data Analysis** — Univariate and bivariate analysis of numerical, ordinal, and categorical features against attrition, backed by statistical tests rather than visuals alone:
- Mann-Whitney U tests for numerical features vs. attrition
- Chi-squared tests + Cramer's V for categorical features
- Spearman correlation for ordinal features

**3. Modeling** — Logistic Regression (baseline) vs. Random Forest, evaluated with ROC-AUC and PR-AUC (accuracy is misleading on an imbalanced target).

## Key Findings
- **Strongest predictors:** `OverTime`, `MonthlyIncome`, `Age`, tenure-related features (`TotalWorkingYears`, `YearsAtCompany`, `YearsInCurrentRole`, `YearsWithCurrManager`), and `JobSatisfaction`
- Employees who left tend to be younger, earn less, have fewer total working years, and work overtime more often
- Tenure features (`YearsAtCompany`, `YearsInCurrentRole`, `YearsWithCurrManager`, `YearsSinceLastPromotion`) are highly correlated with each other — this multicollinearity was handled by trimming redundant features for the linear model

## Results

| Model | ROC-AUC | PR-AUC | Recall (Left) |
|---|---|---|---|
| Logistic Regression | 0.789 | 0.552 | 0.64 |
| Random Forest | 0.779 | 0.410 | 0.26 |

Random Forest has slightly higher overall accuracy, but Logistic Regression catches far more of the employees who actually leave (recall 0.64 vs 0.26) — the more useful model in practice, since missing a departing employee is costlier than a false alarm.

## Tech Stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, SciPy
