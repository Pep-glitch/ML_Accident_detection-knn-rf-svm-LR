# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
data_path = 'D:\ClassSem5\Subjects\ML\Project files\datasets\Datasets\Clean_Encoded_RTA_Dataset.csv'
dataset = pd.read_csv(data_path)
#data = pd.read_csv('Salaries.csv')
X = dataset.drop('Accident_severity', axis=1)
y = dataset['Accident_severity']
# Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Training
rf_classifier.fit(X_train, y_train)

# Prediction
y_pred = rf_classifier.predict(X_test)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Importance of Features
feature_importance = rf_classifier.feature_importances_
print("Feature Importance:\n", feature_importance)
#print(data)
"""
#Select all rows and column 1 from the dataset to x and all rows and column 2 as y.0
x = df.iloc[:, : -1]
y = df.iloc[:, -1:]
"""
# Fitting Random Forest Regression to the dataset
# x = dataset
# x = dataset.drop('Accident_severity', axis=1)
# y = dataset['Accident_severity']
"""
# create regressor object
regressor = RandomForestRegressor(n_estimators=100,
								random_state=0)

# fit the regressor with x and y data
regressor.fit(x, y)

# test the output by changing values
Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))

# Visualising the Random Forest Regression results

# arrange for creating a range of values
# from min value of x to max
# value of x with a difference of 0.01
# between two consecutive values
X_grid = np.arrange(min(x), max(x), 0.01)

# reshape for reshaping the data
# into a len(X_grid)*1 array,
# i.e. to make a column out of the X_grid value
X_grid = X_grid.reshape((len(X_grid), 1))

# Scatter plot for original data
plt.scatter(x, y, color='blue')

# plot predicted data
plt.plot(X_grid, regressor.predict(X_grid),
		color='green')
plt.title('Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
"""

