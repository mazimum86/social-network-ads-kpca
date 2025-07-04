# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 15:08:41 2025
@Author: Chukwuka Chijioke Jerry

Description:
Logistic Regression on the Social Network Ads dataset using Kernel PCA
"""

# ==================== Import Libraries ====================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ==================== Load Dataset ====================

dataset = pd.read_csv('../data/Social_Network_Ads.csv')

# Select features and target
X = dataset.iloc[:, [0, 1]].values  # Age and EstimatedSalary
y = dataset.iloc[:, -1].values      # Purchased

# ==================== Split Train/Test ====================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ==================== Feature Scaling ====================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================== Apply Kernel PCA ====================

kpca = KernelPCA(n_components=2, kernel='rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

# ==================== Train Logistic Regression ====================

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# ==================== Make Predictions ====================

y_pred = classifier.predict(X_test)

# ==================== Evaluation ====================

print("\n=== Accuracy Score ===")
print(f"{accuracy_score(y_test, y_pred):.4f}")

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='g',
    cmap='viridis', cbar=False,
    xticklabels=['Not Purchased', 'Purchased'],
    yticklabels=['Not Purchased', 'Purchased']
)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ==================== Visualization Function ====================

def plot_decision_boundary(X_set, y_set, title):
    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01)
    )
    Z = classifier.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)
    
    plt.contourf(X1, X2, Z, alpha=0.25, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == j, 0], X_set[y_set == j, 1],
            c=ListedColormap(('red', 'green'))(i),
            label=f'Class {j}'
        )
    
    plt.title(title)
    plt.xlabel('KPC 1')
    plt.ylabel('KPC 2')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==================== Plot Decision Boundaries ====================

plot_decision_boundary(X_train, y_train, 'Logistic Regression (Training Set)')
plot_decision_boundary(X_test, y_test, 'Logistic Regression (Test Set)')
