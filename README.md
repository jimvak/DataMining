# Classification Modeling for Stroke Prediction and Spam Detection

This repository contains a machine learning project focused on two supervised classification tasks:

- **Stroke prediction** using structured healthcare data
- **Spam detection** using email/text-based data

The project was developed as part of a university Data Mining course and demonstrates an end-to-end machine learning workflow, including exploratory analysis, data preprocessing, feature preparation, model training, validation, and performance evaluation.

## Project Objectives

The main goals of this project were to:

- apply machine learning techniques to real classification problems
- preprocess and prepare datasets for modeling
- train and compare multiple classification approaches
- evaluate predictive performance using appropriate classification metrics
- identify effective models for different types of data

## Datasets

The project is based on two datasets:

### 1. Stroke Prediction Dataset
A structured dataset containing healthcare-related features used to predict stroke risk.

### 2. Spam Detection Dataset
A dataset used to classify whether a message/email is spam or not spam.

## Methods and Techniques

Across the two tasks, the project includes work on:

- exploratory data analysis
- data preprocessing
- missing-value handling
- feature preparation
- classification modeling
- model validation
- performance comparison

The project helped strengthen practical skills in:

- Python for machine learning
- supervised learning workflows
- structured and text-based data analysis
- classification modeling
- model evaluation using precision, recall, and F1-score

## Tools and Libraries

- Python
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- TensorFlow / Keras
- gensim

## Results

### Stroke Prediction
- Performed exploratory analysis and visualization of the healthcare dataset
- Applied multiple missing-value handling approaches, including column removal, mean imputation, linear regression imputation, and k-nearest neighbors imputation
- Trained and evaluated a Random Forest classifier
- Compared model performance using **precision**, **recall**, and **F1-score**

### Spam Detection
- Converted email text into numerical representations using word embeddings
- Trained a neural-network-based classifier for spam detection
- Evaluated model performance using **precision**, **recall**, and **F1-score**

### Key Takeaways
- Data preprocessing choices had a visible impact on classification performance
- Different machine learning methods were better suited to structured and text-based datasets
- The project strengthened practical skills in classification modeling, preprocessing, embeddings, and model evaluation

## Installation

Install the required libraries with:

```bash
pip install -r requirements.txt
```
## How to Run

1. Clone the repository
2. Install the required libraries using `requirements.txt`
3. Run the scripts inside `healthcare-dataset-stroke-data` for the stroke prediction task
4. Run the script inside `spam_or_not_spam` for the spam detection task

## Repository Structure

## Repository Structure

```text
PROJECT_DM_26_5_2021/
├── healthcare-dataset-stroke-data/
│   ├── healthcare-dataset-stroke-data.csv
│   ├── stroke_dataset_exploration.py
│   ├── stroke_random_forest_mean_imputation.py
│   ├── stroke_random_forest_knn_imputation.py
│   ├── stroke_random_forest_linear_regression_imputation.py
│   └── stroke_random_forest_column_removal.py
├── spam_or_not_spam/
│   ├── spam_or_not_spam.csv
│   └── spam_embedding_neural_network.py
├── Αναφορά.pdf
├── requirements.txt
└── README.md
```

## Project Summary

This project applies machine learning techniques to both structured and text-based classification problems. It combines exploratory analysis, preprocessing, imputation strategies, ensemble learning, embeddings, neural networks, and metric-based evaluation, making it a strong applied data science project for demonstrating junior-level machine learning and data analysis skills.
