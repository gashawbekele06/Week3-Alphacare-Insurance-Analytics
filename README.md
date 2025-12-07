# Week3-Alphacare-Insurance-Analytics

# AlphaCare Insurance Solutions (ACIS) Analytics Project: Risk & Marketing Optimization

## Project Overview

This project serves as the foundational data analysis and predictive modeling effort for **AlphaCare Insurance Solutions (ACIS)** marketing analytics team. The primary goal is to leverage historical car insurance claim data from South Africa to **optimize marketing strategies** and **identify low-risk customer segments** for potential premium reduction, thereby driving new client acquisition.

---

## Business Objectives

AlphaCare Insurance Solutions (ACIS) is committed to developing cutting-edge risk and predictive analytics for car insurance planning and marketing.

- **Risk Mitigation:** Analyze historical claim data to discover **"low-risk" targets**.
- **Marketing Optimization:** Provide insights to tailor insurance products and marketing efforts
- **Reproducibility:** Establish a fully auditable data pipeline using DVC for regulatory compliance and model debugging.

---

## Repository Structure & Key Technologies

The project utilizes a modular structure for maximum reproducibility, auditability, and collaboration.

| Folder/File                             | Purpose                                                                                           | Status/Content                                            |
| :-------------------------------------- | :------------------------------------------------------------------------------------------------ | :-------------------------------------------------------- |
| **`data/`**                             | **Data Versioning** folder. Holds DVC pointer files (`.dvc`) and the working copy of the dataset. | Contains `MachineLearningRating_v3.txt.dvc`               |
| **`data/MachineLearningRating_v3.txt`** | The actual dataset (restored via `dvc checkout`).                                                 | **Data File**                                             |
| **`notebooks/`**                        | Interactive analysis and exploration.                                                             | Contains `eda_analysis.ipynb`                             |
| **`src/`**                              | Modular, reusable Python source code.                                                             | Contains `data_loader.py`, `eda_analysis.py`, `config.py` |
| **`.dvc/`**                             | DVC internal files (cache, config, logs).                                                         | **Ignored by Git**                                        |
| **`.github/`**                          | GitHub Actions workflows for Continuous Integration.                                              | **Automation**                                            |
| **`.gitignore`**                        | Configures files/folders ignored by Git. Includes DVC exceptions.                                 | **Config**                                                |
| **`requirements.txt`**                  | Lists all required Python packages (DVC included).                                                | **Environment**                                           |

---

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
