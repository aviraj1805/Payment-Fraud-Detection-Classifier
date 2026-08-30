# Fraud Detection Classifier

Machine learning models for detecting fraudulent transactions in online payment systems using classification algorithms and comprehensive data analysis.

## Overview

This project implements and compares multiple machine learning classifiers to identify fraudulent transactions with high accuracy. The analysis includes exploratory data analysis, feature engineering, and model evaluation using real-world payment transaction data.

## Features

- Comprehensive exploratory data analysis with correlation analysis and distribution visualizations
- Data preprocessing including one-hot encoding and feature scaling
- Multiple classifier implementations:
  - Decision Tree Classifier
  - Logistic Regression Classifier
- Model evaluation using accuracy, F1-score, and confusion matrices
- Class imbalance handling with balanced class weights
- Train-test split with stratification for better representation

## Dataset

The project uses the Online Fraud dataset (`onlinefraud.csv`) containing transaction records with the following key features:
- Transaction type and amount
- Originator and recipient balance information
- Binary target variable indicating fraud status (isFraud)

## Model Performance

| Model | Accuracy | F1-Score | Notes |
|-------|----------|----------|-------|
| Decision Tree (Tuned) | 99.87% | High | Overfitting detected |
| Logistic Regression (Balanced) | 94.916% | Moderate-High | Better generalization |

## Installation

### Requirements
- Python 3.7+
- pandas
- scikit-learn
- matplotlib
- seaborn

### Setup

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/fraud-detection-classifier.git
   cd fraud-detection-classifier
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset is in the project directory
   ```bash
   # Place onlinefraud.csv in the project root
   ```

## Usage

Run the Jupyter notebook to execute the complete analysis:

```bash
jupyter notebook main.ipynb
```

The notebook performs the following steps:
1. Data loading and initial exploration
2. Statistical analysis and visualization
3. Feature preprocessing and engineering
4. Model training and evaluation
5. Performance comparison and analysis

## Project Structure

```
fraud-detection-classifier/
├── main.ipynb
├── onlinefraud.csv
├── requirements.txt
└── README.md
```

## Methodology

### 1. Exploratory Data Analysis
- Summary statistics and data distribution analysis
- Correlation heatmaps between numerical features
- Box plots comparing feature distributions by fraud status
- Scatter plots and pair plots for multi-dimensional analysis

### 2. Data Preprocessing
- Identification and handling of missing values
- One-hot encoding of categorical features (transaction type)
- Removal of irrelevant features (nameOrig, nameDest)
- Feature scaling using StandardScaler for logistic regression

### 3. Model Development
- Train-test split with 80-20 ratio and stratification
- Grid search for hyperparameter tuning
- Cross-validation for robust model selection
- Class weight balancing to address data imbalance

### 4. Evaluation Metrics
- Accuracy score
- F1-score for imbalanced dataset performance
- Confusion matrix for detailed error analysis
- Overfitting detection through train-test comparison

## Key Findings

- The Decision Tree classifier achieved high accuracy but showed signs of overfitting with nearly perfect train accuracy
- The Logistic Regression model provided better generalization with balanced class weights
- Class imbalance in the dataset required careful handling through stratified splitting and class weight adjustment
- Feature scaling significantly improved model performance for distance-based algorithms

## Results

The Logistic Regression classifier with balanced class weights demonstrated the best balance between accuracy and generalization, achieving 94.916% accuracy on the test set with better real-world applicability compared to the overfitted decision tree model.

## Future Improvements

- Ensemble methods (Random Forest, Gradient Boosting)
- Additional feature engineering and selection techniques
- Cross-validation optimization
- Deployment as a web service
- Real-time fraud detection pipeline

## License

This project is available under the MIT License. See LICENSE file for details.

## Contact

For questions or contributions, please open an issue or submit a pull request.
