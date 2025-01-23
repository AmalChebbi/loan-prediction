# Loan Default Prediction using Neural Networks

This project implements a binary classification model to predict the likelihood of loan defaults. The dataset is preprocessed, trained, and evaluated using PyTorch. The project demonstrates a complete machine learning pipeline, from preprocessing raw data to generating predictions on the test dataset.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Model Architecture](#model-architecture)
4. [Pipeline](#pipeline)
5. [Usage](#usage)
6. [Results](#results)
7. [Acknowledgments](#acknowledgments)

---

## Project Overview

The goal of this project is to predict the `bad_flag` (loan default flag) for borrowers based on loan data. The project includes:
- Data preprocessing, including handling categorical and numerical features.
- Scaling and encoding data for machine learning.
- Building a neural network for binary classification.
- Training, evaluating, and making predictions on a test set.

---

## Technologies Used

- **Python**: Core programming language.
- **PyTorch**: Deep learning framework for model training and evaluation.
- **Pandas**: Data manipulation and analysis.
- **Scikit-learn**: Data preprocessing and evaluation metrics.
- **Matplotlib/Seaborn**: Data visualization.

---

## Model Architecture

The neural network consists of:
1. Input Layer: Matches the number of features in the training dataset.
2. Hidden Layers:
   - 64 units with ReLU activation.
   - 32 units with ReLU activation.
3. Output Layer:
   - 1 unit with a sigmoid activation function for binary classification.

Loss Function: Binary Cross-Entropy Loss (BCELoss)  
Optimizer: Adam Optimizer (learning rate = 0.001)

---

## Pipeline

### 1. Data Preprocessing
- Missing values handled (median for numeric, "Unknown" for categorical).
- Encoding categorical columns using one-hot encoding.
- Scaling numerical features to zero mean and unit variance.

### 2. Model Training
- A PyTorch-based neural network is trained for 20 epochs.
- Batch size: 64
- Metrics: Training and validation loss, accuracy.

### 3. Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **AUC-ROC**

### 4. Inference
- Predictions are generated on the test dataset.
- Results are saved as a CSV file (`test_predictions.csv`).

---

## Usage

### Prerequisites
1. Python 3.8+
2. Libraries: `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

### Steps
1. Clone the repository and navigate to the project folder.
2. Run the Jupyter Notebook (`Project.ipynb`).
3. Test predictions will be saved in `test_predictions.csv`.

---

## Results

### Validation Metrics:
- **Accuracy**: 90.25%
- **Precision**: 97.10%
- **Recall**: 82.97%
- **F1-Score**: 89.48%
- **AUC-ROC**: 95.04%

The model demonstrates strong performance in classifying loan defaults with high precision and recall.

---

## Acknowledgments

Thank you to everyone who reviews and evaluates this work.
