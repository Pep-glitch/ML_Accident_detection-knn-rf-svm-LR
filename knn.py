# Import necessary modules
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_path = 'D:\ClassSem5\Subjects\ML\Project files\datasets\Datasets\Clean_Encoded_RTA_Dataset.csv'
dataset = pd.read_csv(data_path)

# Create feature and target arrays
X = dataset.drop('Accident_severity', axis=1)
y = dataset['Accident_severity']

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
"""
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
"""
# Fitting the model
K_neighbor = KNeighborsClassifier(n_neighbors=8)
K_neighbor.fit(X_train, y_train)
# Prediction
y_pred = K_neighbor.predict(X_test)

# Loop over K values
"""
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()
"""

#print(X_test.shape)

# Evaluation
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
print(f"F1 Score: {f1}")
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
print(f"Recall:{recall}")
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
print(f"Precision:{precision}")
# print(f"Specificity:{specificity}")
print(classification_report(y_test, y_pred))
