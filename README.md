# 🧠 Machine Learning Project: Model Selection for KNN

This project focuses on the **evaluation** phase of the machine learning life cycle. You'll use **model selection** to identify the optimal K-Nearest Neighbors (KNN) classifier for a classification problem, then evaluate its performance using several metrics.

### 📌 Project Goals

* Define a classification problem using a dataset
* Create labeled examples
* Split data into **training** and **test** sets
* Use **grid search** to select the best value for `k` (number of neighbors)
* Train and evaluate a **KNN classifier**
* Generate a **confusion matrix**
* Plot a **precision-recall curve**

> ⚠️ Note: The `predict_proba()` method in KNN doesn’t return true probabilities, because KNN is a non-probabilistic algorithm. It calculates the proportion of neighbors belonging to each class to simulate class "probabilities". Still, this is useful for evaluating metrics like precision and recall.

---

### 🛠️ Steps

#### 1. Build the DataFrame and Define the ML Problem

```python
import pandas as pd
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target
df = pd.DataFrame(X)
df['label'] = y
```

#### 2. Create Labeled Examples

```python
# Already included above; y is our label
```

#### 3. Split the Data

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 4. Perform Grid Search for Best `k`

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param_grid = {'n_neighbors': list(range(1, 11))}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

best_k = grid.best_params_['n_neighbors']
print(f"Best k: {best_k}")
```

#### 5. Fit and Predict with the Optimal KNN Model

```python
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

#### 6. Evaluate Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

#### 7. Confusion Matrix & Precision-Recall Curve

```python
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Precision-Recall Curve (binary example - e.g., classifying digit '1' vs not '1')
y_test_bin = (y_test == 1).astype(int)
y_scores = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test_bin, y_scores)

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()
```
