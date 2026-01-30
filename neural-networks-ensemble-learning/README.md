# Neural Networks and Ensemble Learning with Python

This project studies supervised multi-class classification using neural networks and ensemble learning.
It includes (1) a custom implementation of a multi-class perceptron, (2) an MLP classifier with and without feature normalization, and (3) a bagging (bootstrap aggregating) approach with majority voting.

## What’s inside
- **Stratified train/test split (2/3 – 1/3)** implemented manually to ensure class balance
- **Multi-class Perceptron (from scratch)**: weight matrix update, label encoding, training loop
- **MLPClassifier (scikit-learn)** with comparison:
  - without normalization
  - with normalization (**StandardScaler**)
- **Model evaluation**:
  - confusion matrix
  - overall accuracy
  - per-class precision and recall
- **Bagging of MLPs (from scratch)**:
  - K bootstrap samples
  - train one MLP per bootstrap sample
  - aggregate predictions by **majority vote** (mode)

## Datasets
Experiments are run on multiple benchmark datasets:
- Iris
- Glass
- Breast Cancer Wisconsin
- Lsun
- Wave

## Tech stack
- Python
- Jupyter Notebook
- NumPy, Pandas
- scikit-learn (MLPClassifier, StandardScaler, metrics)
- Matplotlib

## How to run
1. Open the notebook:
   - `mlp-bagging-classification.ipynb`
2. Place datasets (e.g., `iris.txt`, `glass.txt`, ...) in the same folder (or update file paths in the notebook).
3. Run all cells from top to bottom.

## Notes / Results
- The notebook compares MLP performance **with vs without normalization**.
- Bagging is used to improve robustness via variance reduction.
- Final performance metrics are displayed in the notebook outputs (confusion matrix + accuracy/precision/recall).

