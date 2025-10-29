# train_model.py
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import cross_val_score
os.makedirs("outputs", exist_ok=True)
X = np.load("outputs/X_features.npy")
y = np.load("outputs/y_labels.npy")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "outputs/scaler.joblib")

# Train classifier (Random Forest as baseline)
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)
print(classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["GOOD","BAD"], yticklabels=["GOOD","BAD"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (acc={acc:.3f})")
plt.savefig("outputs/figures/confusion_matrix.png")
plt.close()

# Save model
joblib.dump(clf, "outputs/rf_model.joblib")
print("Model and scaler saved to outputs/")

scores = cross_val_score(clf, scaler.transform(X), y, cv=5, scoring='accuracy', n_jobs=-1)
print("5-fold CV accuracy:", scores.mean(), scores.std())

