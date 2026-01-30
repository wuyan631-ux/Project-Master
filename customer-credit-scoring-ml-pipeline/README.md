# Customer Credit Scoring – Supervised ML Pipeline

This project focuses on building an end-to-end supervised machine learning pipeline for **customer credit scoring**.
The objective is to predict customer solvency while following a rigorous data science methodology, from data preprocessing to model selection, evaluation, and deployment-ready pipelines.

## Problem Context
Customer credit scoring is a binary classification task with strong business constraints.
In this project, model evaluation prioritizes **precision** (to reduce false positive approvals) alongside accuracy.

## Workflow Overview
The project implements a complete and reusable ML workflow:
1. Data preprocessing and train/test split
2. Feature normalization and dimensionality reduction
3. Supervised model training and comparison
4. Feature importance analysis and selection
5. Hyperparameter tuning
6. Pipeline construction and serialization
7. Robust model comparison via cross-validation

## Models Evaluated
Multiple supervised learning algorithms are compared:
- Decision Tree (CART)
- k-Nearest Neighbors (KNN)
- Multi-layer Perceptron (MLP)
- Bagging
- AdaBoost
- Random Forest

All experiments ensure reproducibility using fixed random states.

## Evaluation Strategy
Model performance is evaluated using:
- Confusion matrix
- Accuracy
- Precision (business-driven metric)
- ROC curve and AUC
- Custom business score:  
  **(Accuracy + Precision) / 2**
- 10-fold cross-validation with execution time analysis

## Feature Engineering & Selection
- Feature normalization using `StandardScaler`
- Dimensionality reduction using PCA
- Feature importance estimation via Random Forest
- Selection of an optimal subset of features based on performance evolution

## Hyperparameter Optimization
- GridSearchCV applied to the best-performing model
- Optimization based on the custom business score
- Stratified cross-validation for robust tuning

## ML Pipeline & Deployment
- Automated ML pipeline using `scikit-learn Pipeline`
- End-to-end orchestration of preprocessing, modeling, and selection steps
- Final pipeline serialized using `pickle` for deployment or scoring reuse

## Code Structure
- `credit_scoring_supervised_learning_pipeline.ipynb`  
  Main notebook illustrating the full experimental workflow.
- `ml_pipeline_utils.py`  
  Reusable utility functions for training, evaluation, feature selection, tuning, cross-validation, and pipeline orchestration.

## Tech Stack
- Python
- NumPy, Pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook

## Key Takeaways
- End-to-end ML pipeline design with business-oriented evaluation
- Structured and reusable code suitable for real-world ML projects
- Clear separation between experimentation (notebook) and engineering logic (utils)

