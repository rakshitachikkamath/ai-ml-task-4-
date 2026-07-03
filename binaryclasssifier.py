# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import dataset
from sklearn.datasets import load_breast_cancer
# Split dataset
from sklearn.model_selection import train_test_split

# Standardize features
from sklearn.preprocessing import StandardScaler

# Logistic Regression model
from sklearn.linear_model import LogisticRegression

# Evaluation metrics
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score
)

# -------------------------------------------------
# Step 1: Load Dataset
# -------------------------------------------------
data = load_breast_cancer()

# Create DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df["target"] = data.target

print("First 5 rows:")
print(df.head())

# -------------------------------------------------
# Step 2: Split Features and Target
# -------------------------------------------------
X = df.drop("target", axis=1)
y = df["target"]

# -------------------------------------------------
# Step 3: Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------------------------
# Step 4: Standardize Features
# -------------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------
# Step 5: Train Logistic Regression Model
# -------------------------------------------------
model = LogisticRegression()

model.fit(X_train, y_train)

# -------------------------------------------------
# Step 6: Make Predictions
# -------------------------------------------------
y_pred = model.predict(X_test)

# Probability predictions
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------------------------
# Step 7: Evaluation
# -------------------------------------------------
print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# -------------------------------------------------
# Step 8: Confusion Matrix
# -------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()

# -------------------------------------------------
# Step 9: ROC Curve
# -------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7,5))

plt.plot(fpr, tpr, label="ROC Curve")

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# -------------------------------------------------
# Step 10: Threshold Tuning
# -------------------------------------------------
custom_threshold = 0.6

custom_prediction = (y_prob >= custom_threshold).astype(int)

print("\nUsing Threshold =", custom_threshold)

print(classification_report(y_test, custom_prediction))

# -------------------------------------------------
# Step 11: Sigmoid Function
# -------------------------------------------------
x = np.linspace(-10,10,100)

sigmoid = 1/(1+np.exp(-x))

plt.figure(figsize=(7,5))

plt.plot(x, sigmoid)

plt.title("Sigmoid Function")

plt.xlabel("Input")

plt.ylabel("Probability")

plt.grid()

plt.show()