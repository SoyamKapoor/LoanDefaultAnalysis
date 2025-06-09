# Loan Default Prediction

## Overview

This Python project predicts loan defaults using the Loans Full Schema dataset. It implements Support Vector Machine (SVM) and LightGBM models, with SMOTE to address class imbalance. Leveraging pandas, scikit-learn, lightgbm, matplotlib, and seaborn, the system identifies key predictors (annual income, debt-to-income ratio, loan amount) and prioritizes recall to minimize false negatives.

---

## Features

- **Data Preprocessing:** Filters loan statuses, handles missing values, applies one-hot encoding, and normalizes features using StandardScaler.
- **Class Imbalance Handling:** Uses SMOTE to oversample the minority class (defaults).
- **Model Training:** Implements SVM and LightGBM for classification, with LightGBM showing superior performance.
- **Evaluation:** Assesses models using precision, recall, F1-score, confusion matrices, and ROC-AUC, focusing on reducing false negatives.
- **Feature Analysis:** Highlights annual income, debt-to-income ratio, and loan amount as key predictors.
- **Visualization:** Plots class distribution and ROC curves for interpretability.

---

## Usage

1. **Load Dataset:** Imports Loans Full Schema dataset using pandas.  
2. **Preprocess Data:** Filters statuses, handles missing values, encodes categorical variables, and normalizes features.  
3. **Exploratory Analysis:** Visualizes class distribution with seaborn to identify imbalance.  
4. **Train Models:** Trains SVM and LightGBM models, using SMOTE for balanced training data.  
5. **Evaluate & Visualize:** Computes performance metrics and visualizes results with ROC curves and confusion matrices.  
6. **Recommendations:** Suggests verifying key features (annual income, debt-to-income ratio, loan amount) before loan approval.

---

## Dependencies

- Python 3.11  
- pandas  
- numpy  
- scikit-learn  
- lightgbm  
- imblearn  
- matplotlib  
- seaborn  

---

## Project Structure

- `loan_default_prediction.ipynb`: Main implementation notebook  
- `README.md`: Project documentation  

---

## Results

LightGBM outperforms SVM with strong predictive power and interpretability. Key predictors include annual income, debt-to-income ratio, and loan amount. The system supports financial institutions by reducing false negatives, aiding in risk assessment and loan approval decisions.

---

## Future Improvements

- Incorporate hyperparameter tuning for SVM and LightGBM to enhance performance.  
- Explore additional models like XGBoost or neural networks for comparison.  
- Integrate real-time data updates for dynamic risk assessment.

---

## Top Skills Demonstrated

- Machine Learning  
- Data Preprocessing  
- Data Visualization  
- Python Programming  
- Class Imbalance Handling  

---

## Acknowledgments

- Loans Full Schema dataset for providing comprehensive data  
- scikit-learn and lightgbm for robust machine learning tools  
- imblearn for SMOTE implementation  
- matplotlib and seaborn for visualization capabilities
