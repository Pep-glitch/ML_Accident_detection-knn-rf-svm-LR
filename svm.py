# Load the important packages
import warnings

from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning


# Load the datasets
data_path = 'D:\ClassSem5\Subjects\ML\Project files\datasets\Datasets\Clean_Encoded_RTA_Dataset.csv'
dataset = pd.read_csv(data_path)
X = dataset.drop('Accident_severity', axis=1)
# Target Column
y = dataset['Accident_severity']

# Splitting the data set into training abd testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# SVM classifier
clf = SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)

# Predicting
y_pred = clf.predict(X_test)

# Evaluating the model
# Calculating Specificity using Confusion Matrix
"""
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)
"""
# Other parameters
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
print(f"F1 Score: {f1}")
recall = recall_score(y_test, y_pred,average='weighted', zero_division=1)
print(f"Recall:{recall}")
precision = precision_score(y_test, y_pred,average='weighted', zero_division=1)
print(f"Precision:{precision}")
# print(f"Specificity:{specificity}")
print(classification_report(y_test, y_pred))
