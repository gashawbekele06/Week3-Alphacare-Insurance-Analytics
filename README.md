# Week3-Alphacare-Insurance-Analytics

# AlphaCare Insurance Solutions (ACIS) Analytics Project: Risk & Marketing Optimization

## Project Overview

This project serves as the foundational data analysis and predictive modeling effort for **AlphaCare Insurance Solutions (ACIS)** marketing analytics team. The primary goal is to leverage historical car insurance claim data from South Africa to **optimize marketing strategies** and **identify low-risk customer segments** for potential premium reduction, thereby driving new client acquisition.

---

## Project Structure

WEEK3-ALPHACARE-INSURANCE-ANALYTICS/
├─ .dvc/
│ ├─ cache/
│ │ └─ files/
│ │ └─ md5/
│ │ └─ f6/
│ │ └─ b7009b68ae21372b7deca9307fbb23 # example cached object directory
│ ├─ tmp/
│ └─ config # DVC config for remotes, etc.
├─ .github/ # (optional) GitHub workflows and settings
├─ data/
│ ├─ MachineLearningRating_v3.txt # dataset (tracked via DVC)
│ └─ MachineLearningRating_v3.txt.dvc # DVC pointer file for the dataset
├─ notebooks/
│ ├─ **init**.py
│ └─ eda_analysis.ipynb # exploratory data analysis notebook
├─ src/
│ ├─ **pycache**/ # Python bytecode cache (auto-generated)
│ ├─ **init**.py
│ ├─ config.py # central config (paths, params)
│ ├─ data_loader.py # data loading utilities
│ └─ eda_analysis.py # script version of EDA
├─ .dvcignore
├─ .gitignore
├─ README.md
└─ requirements.txt # Python dependencies

---

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
