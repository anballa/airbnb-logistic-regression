# Airbnb Logistic Regression Model

This repository contains the code and data used to build and evaluate a logistic regression model to predict Airbnb listing quality. The project demonstrates data preprocessing, feature selection, hyperparameter tuning, model training, evaluation, and saving/loading models with Python and scikit-learn.

## Project Overview

- Dataset: Preprocessed Airbnb listings data ('airbnbData_train.csv')
- Goal: Predict the 'great_quality' label using logistic regression
- Key steps:
  - Feature selection using 'SelectKBest' with ANOVA F-test
  - Model tuning with grid search for the best regularization parameter 'C'
  - Model evaluation using Precision-Recall and ROC curves
  - Model serialization using 'pickle'

## Files

- 'airbnbData_train.csv' - The training dataset with preprocessing completed
- 'logistic_model.pkl' - Saved logistic regression model with best hyperparameters
- 'README.md' - This file

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- seaborn
- matplotlib
