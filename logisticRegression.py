import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.exceptions import UndefinedMetricWarning

# Load the datasets
data_path = 'D:\ClassSem5\Subjects\ML\Project files\datasets\Datasets\Clean_Encoded_RTA_Dataset.csv'
dataset = pd.read_csv(data_path)

# Logistic Regression model
logistic_regression_model = LogisticRegression()

# Create feature and target arrays
X = dataset.drop('Accident_severity', axis=1)
y = dataset['Accident_severity']

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
logistic_regression_model.fit(X_train, y_train)

# Prediction
y_pred = logistic_regression_model.predict(X_test)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Regularization
# logistic_regression_model = LogisticRegression(penalty='l2', C=1.0)

"""
# input
x = dataset.iloc[:, [2, 3]].values

# output
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(
	x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)

print (xtrain[0:10, :])

from sklearn.linear_model import LogisticRegression

# predictions
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)

#test on training data
y_pred = classifier.predict(xtest)

# performance check
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ytest, y_pred)
print ("Confusion Matrix : \n", cm)

from sklearn.metrics import accuracy_score

print ("Accuracy : ", accuracy_score(ytest, y_pred))
"""
