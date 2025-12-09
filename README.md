# Week 3: AlphaCare Insurance – Risk-Based Pricing & Predictive Analytics

**End-to-End Insurance Analytics Pipeline**  
**Author**: Gashaw Bekele  
**Date**: December 9, 2025  
**GitHub**: https://github.com/gashawbekele06/Week3-Alphacare-Insurance-Analytics

---

## Project Goal

Build a **dynamic, risk-based car insurance pricing system** for AlphaCare Insurance Solutions using historical claims data (Feb 2014 – Aug 2015) to:

- Identify low-risk customer segments
- Reduce loss ratio
- Increase profitability
- Ensure full auditability and reproducibility

---

## Project Objectives

1. Perform comprehensive EDA & statistical hypothesis testing
2. Implement reproducible data pipeline using **DVC**
3. Validate risk drivers via A/B testing
4. Build predictive models for claim severity and probability
5. Deliver **risk-based premium formula** with SHAP interpretability

## Project Folder Structure

```text
Week3-Alphacare-Insurance-Analytics/
├── .dvc/
├── data/
│   └── MachineLearningRating_v3.txt.dvc
├── notebooks/
│   ├── hypothesis_testing.ipynb
│   └── predictive_modeling.ipynb
├── src/
│   ├── data_loader.py
│   ├── hypothesis_testing.py
│   └── predictive_modeling.py
├── .dvcignore
├── .gitignore
├── requirements.txt
└── README.md

```

## Data Sources

The project utilizes historical car insurance claim data spanning **February 2014 to August 2015**.

| Data Column Group   | Examples                                                                                 |
| :------------------ | :--------------------------------------------------------------------------------------- |
| **Policy Details**  | `PolicyID`, `UnderwrittenCoverID`, `TransactionDate`, `TransactionMonth`                 |
| **Client/Owner**    | `Citizenship`, `Gender`, `MaritalStatus`, `Language`, `PostalCode`, `Province`           |
| **Vehicle Details** | `Make`, `Model`, `RegistrationYear`, `Cylinders`, `Cubiccapacity`, `CustomValueEstimate` |
| **Plan & Finance**  | `TotalPremium`, `TotalClaims`, `SumInsured`, `ExcessSelected`, `CoverType`               |

---

## Tasks Completed

| Task       | Description               | Key Output                                                    |
| ---------- | ------------------------- | ------------------------------------------------------------- |
| **Task 1** | Exploratory Data Analysis | Loss ratio by province, gender, vehicle type; temporal trends |
| **Task 2** | Data Version Control      | DVC pipeline with local remote; `dvc pull` restores all data  |
| **Task 3** | Hypothesis Testing        | **All 4 null hypotheses rejected** (p < 0.01)                 |
| **Task 4** | Predictive Modeling       | XGBoost severity model (RMSE: R 4,821, R²: 0.82) + SHAP       |

---

## Key Findings

| Insight                        | Evidence                    | Business Implication        |
| ------------------------------ | --------------------------- | --------------------------- |
| Gauteng = highest risk         | Highest claim frequency     | **+15% premium loading**    |
| Older vehicles = higher claims | +R 1,200 per year (SHAP)    | **Age-based pricing**       |
| Luxury brands = severe claims  | BMW/Mercedes top SHAP       | **+20% brand loading**      |
| Females = lower risk           | Lower claim rate (p < 0.01) | **6–8% gender discount**    |
| Postcode variation             | Significant (p < 0.01)      | **Enable granular pricing** |

---

## Reproducibility and Setup Guide

This project is versioned using **Git** for code and **DVC** for the large dataset (`MachineLearningRating_v3.txt`). To set up the environment and retrieve the data, follow these steps:

### How to Reproduce This Project (Exact Steps)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Week3-Alphacare-Insurance-Analytics.git
cd Week3-Alphacare-Insurance-Analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download real data & notebooks (critical!)
dvc pull

# 4. Run the analysis
python src/hypothesis_testing.py
python src/predictive_modeling.py

# 5. Open notebooks
jupyter lab notebooks/hypothesis_testing.ipynb
jupyter lab notebooks/predictive_modeling.ipynb
```
