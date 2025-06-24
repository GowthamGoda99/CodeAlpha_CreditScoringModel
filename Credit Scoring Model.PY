import pandas as pd

# Load the uploaded CSV file
df = pd.read_csv("/content/Loan_Default.csv")

# Show shape and a few rows
print("✅ Dataset loaded! Shape:", df.shape)
df.head()

# Display column names to verify target column
print("📌 Available Columns:", df.columns.tolist())

# Drop rows with missing values
df = df.dropna()
print("🧹 Missing values removed. New shape:", df.shape)

from sklearn.preprocessing import LabelEncoder

# Label encode all object-type (string) columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print("🔁 Categorical columns encoded.")

# Clean column names: lowercase and replace spaces with underscores
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("🧾 Cleaned Column Names:", df.columns.tolist())

# Set the correct name of your target column
target_column = "status"  # ← 'status' is the column we will predict

# Ensure the column exists
if target_column not in df.columns:
    raise ValueError(f"❌ Target column '{target_column}' not found in the dataset!")

# Split features and target
X = df.drop(target_column, axis=1)
y = df[target_column]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
print("📊 Features scaled.")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("📦 Data split complete.")
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Only try y_proba if there are 2 classes
if len(model.classes_) > 1:
    y_proba = model.predict_proba(X_test)[:, 1]
else:
    y_proba = [0] * len(X_test)

print("🌲 Model training complete.")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

print("📊 Evaluation Metrics:")
print("Accuracy      :", accuracy_score(y_test, y_pred))
print("Precision     :", precision_score(y_test, y_pred))
print("Recall        :", recall_score(y_test, y_pred))
print("F1 Score      :", f1_score(y_test, y_pred))
print("ROC-AUC Score :", roc_auc_score(y_test, y_proba))

print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("🔷 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("🔷 ROC Curve")
plt.legend()
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get feature names and importances
feature_names = df.drop(target_column, axis=1).columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Set up plot
plt.figure(figsize=(10, 5))

# Check if the model learned anything
if importances.sum() == 0:
    # Plot empty chart
    plt.barh([], [])
    plt.title("🔷 Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.text(0.5, 0.5, "⚠️ No feature importance assigned by the model.",
             ha='center', va='center', fontsize=12, color='red', transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()
else:
    # Plot real importances
    sns.barplot(x=importances[indices], y=feature_names[indices])
    plt.title("🔷 Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
