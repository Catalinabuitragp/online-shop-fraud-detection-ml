# Fraud Detection in Online Shops Using Machine Learning Classification Models

## Overview

This project develops and evaluates machine learning models to detect fraudulent online shops using website characteristics, SSL certificate information, payment methods, and reputation indicators.

The project follows a complete machine learning workflow, including:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- One-Hot Encoding
- Correlation analysis
- Feature selection
- Model training
- Model evaluation

Three classification models were developed and compared:

- Dummy Classifier (Baseline)
- Decision Tree
- Random Forest

Among the evaluated models, the **Random Forest** achieved the highest predictive performance with an accuracy of **93.86%**.

---

## Dataset

**Dataset:** Fraudulent Online Shops Dataset

The dataset contains information describing both legitimate and fraudulent online shops.

### Dataset Characteristics

- **1,140 observations**
- **26 original features**
- **72 processed features** after preprocessing and feature engineering

The dataset includes information related to:

- Domain characteristics
- SSL certificates
- Payment methods
- Website reputation
- Domain registration
- TrustPilot information

---

## Technologies

The project was developed in **Python** using the following libraries:

- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Catalinabuitragp/online-shop-fraud-detection-ml.git
```

### 2. Navigate to the project folder

```bash
cd online-shop-fraud-detection-ml
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python src/fraud_detection.py
```

---

## Machine Learning Workflow

The project follows the following pipeline:

1. Load the dataset
2. Exploratory Data Analysis
3. Data preprocessing
4. Feature engineering
5. One-Hot Encoding
6. Correlation analysis
7. Feature selection
8. Train-test split
9. Dummy Classifier
10. Decision Tree
11. Random Forest
12. Model comparison

---

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|--------|---------:|----------:|-------:|---------:|
| Dummy Classifier | 50.88% | 50.88% | 100.00% | 67.44% |
| Decision Tree | 92.54% | 91.60% | 93.97% | 92.77% |
| **Random Forest** | **93.86%** | **93.97%** | **93.97%** | **93.97%** |

The Random Forest provided the best overall performance while reducing dependence on a single predictor and distributing feature importance across multiple fraud indicators.

---

## Repository Structure

```
online-shop-fraud-detection-ml/
│
├── data/
│   └── Fraudulent_online_shops_dataset.csv
│
├── src/
│   └── fraud_detection.py
│
├── figures/
│   ├── class_distribution.png
│   ├── decision_tree.png
│   └── feature_importance.png
│
├── report/
│   └── Fraud_Detection_Report.pdf
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## References

Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

Géron, A. (2023). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3rd ed.).

Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research.

---

## Author

**Catalina Buitrago Torres**

University of the Potomac

Course: Artificial Intelligence

Summer 2026

---

## Source Code

The complete Python implementation developed for this project, including data preprocessing, feature engineering, model training, and model evaluation, is available in this GitHub repository.
