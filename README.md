# Week3-Alphacare-Insurance-Analytics

# AlphaCare Insurance Solutions (ACIS) Analytics Project: Risk & Marketing Optimization

## Project Overview

This project serves as the foundational data analysis and predictive modeling effort for **AlphaCare Insurance Solutions (ACIS)** marketing analytics team. The primary goal is to leverage historical car insurance claim data from South Africa to **optimize marketing strategies** and **identify low-risk customer segments** for potential premium reduction, thereby driving new client acquisition.

---

## 📁 Project Folder Structure

```text
WEEK3-ALPHACARE-INSURANCE-ANALYTICS/
├─ .dvc/                     # DVC configuration and cache directory
├─ .github/                  # GitHub workflows or settings
├─ data/                     # Data files (tracked via DVC)
├─ notebooks/                # Jupyter notebooks for analysis
│  ├─ __init__.py
│  └─ eda_analysis.ipynb     # Exploratory Data Analysis notebook
├─ src/                      # Source code for the project
│  ├─ __pycache__/           # Python bytecode cache
│  ├─ __init__.py
│  ├─ config.py              # Configuration settings
│  ├─ data_loader.py         # Data loading utilities
│  └─ eda_analysis.py        # EDA script
├─ .dvcignore                # Ignore patterns for DVC
├─ .gitignore                # Ignore patterns for Git
├─ README.md                 # Project documentation
└─ requirements.txt          # Python dependencies

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

## Reproducibility and Setup Guide

This project is versioned using **Git** for code and **DVC** for the large dataset (`MachineLearningRating_v3.txt`). To set up the environment and retrieve the data, follow these steps:

### Environment and Code Setup

```bash
# Clone the repository
git clone [https://github.com/gashawbekele06/Week3-Alphacare-Insurance-Analytics.git](https://github.com/gashawbekele06/Week3-Alphacare-Insurance-Analytics.git)
cd Week3-Alphacare-Insurance-Analytics

# Setup Virtual Environment and Install Dependencies
python -m venv venv
source venv/bin/activate  # Use 'venv\Scripts\activate' on Windows
pip install -r requirements.txt

#
```
