# ai-ml-task-4-
# Logistic Regression Classification using Scikit-learn

## Project Overview

This project demonstrates how to build a **binary classification model** using **Logistic Regression** in Python. The model is trained on the **Breast Cancer Wisconsin Dataset** to predict whether a tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)**.

The project includes data preprocessing, model training, prediction, evaluation using multiple metrics, threshold tuning, and visualization of the sigmoid function and ROC curve.

---

# Objective

* Build a binary classification model using Logistic Regression.
* Split the dataset into training and testing sets.
* Standardize the feature values.
* Train the Logistic Regression model.
* Evaluate the model using multiple performance metrics.
* Understand the sigmoid function and threshold tuning.

---

# Technologies Used

* Python 3.x
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

# Project Structure

```text
Logistic_Regression_Project/
│
├── logistic_regression.py
├── README.md
└── requirements.txt
```

---

# Dataset

**Dataset Name:** Breast Cancer Wisconsin Dataset

The dataset is available directly from Scikit-learn, so no separate download is required.

Number of Samples: **569**

Number of Features: **30**

Target Classes:

* 0 → Malignant
* 1 → Benign

---

# Installation

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Or install using:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Execute the Python file:

```bash
python logistic_regression.py
```

---

# Project Workflow

1. Load the Breast Cancer Wisconsin Dataset.
2. Convert the dataset into a Pandas DataFrame.
3. Separate features and target values.
4. Split the dataset into training and testing sets.
5. Standardize the feature values using `StandardScaler`.
6. Train a Logistic Regression model.
7. Predict class labels and probabilities.
8. Evaluate the model using multiple metrics.
9. Plot the Confusion Matrix.
10. Plot the ROC Curve.
11. Perform threshold tuning.
12. Visualize the Sigmoid Function.

---

# Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve
* ROC-AUC Score

---

# Visualizations

The project generates the following plots:

* Confusion Matrix
* ROC Curve
* Sigmoid Function

These visualizations help understand the model's performance and how Logistic Regression converts predictions into probabilities.

---

# Key Concepts

* Binary Classification
* Logistic Regression
* Sigmoid Function
* Train-Test Split
* Feature Scaling
* Standardization
* Confusion Matrix
* Precision
* Recall
* F1-Score
* ROC Curve
* ROC-AUC Score
* Threshold Tuning

---

# Expected Output

The program displays:

* First five rows of the dataset
* Classification Report
* Precision Score
* Recall Score
* ROC-AUC Score
* Confusion Matrix
* ROC Curve
* Sigmoid Function Graph
* Threshold Tuning Results

---

# Learning Outcomes

After completing this project, you will understand:

* How Logistic Regression performs binary classification.
* How to preprocess data before training a model.
* Why feature scaling is important.
* How to train and evaluate a classification model.
* How to interpret Precision, Recall, and F1-Score.
* How ROC Curve and ROC-AUC measure classifier performance.
* How changing the classification threshold affects predictions.
* The role of the Sigmoid Function in Logistic Regression.

---

# Future Improvements

* Perform hyperparameter tuning using GridSearchCV.
* Add cross-validation for better model evaluation.
* Compare Logistic Regression with Decision Tree and Random Forest classifiers.
* Save the trained model using Pickle or Joblib.
* Build a web application using Flask or Streamlit for predictions.

# Author

**Rakshita**

This project was developed as part of a Machine Learning learning task to understand binary classification using Logistic Regression with Scikit-learn.
