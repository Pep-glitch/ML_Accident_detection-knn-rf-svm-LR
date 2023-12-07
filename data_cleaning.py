from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
# file path declaration
file_path = 'D:\ClassSem5\Subjects\ML\Project files\datasets\Datasets\RTA_Dataset.csv'
# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)
# Display the first few rows of the DataFrame to check if it's loaded correctly
print(df.head())
# print("hello")
# df.describe()
# imputing missing values/knn prediction technique
# imputer = KNNImputer(n_neighbors=2)
# df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
# imputing using mean
print(df.isnull().sum())
# Filling missing values using mean
# df.fillna(df.mean(), inplace=True)
print("After imputation")
print(df.head())
print("missing values after imputation")
print(df.isnull().sum())
clean_data_path = 'D:\ClassSem5\Subjects\ML\Project files\datasets\Datasets\Cleaned_RTA_Dataset.csv'
# df.to_csv(clean_data_path, index=False)
print("Data Cleaning Stage 1")
# Using the stage 1 cleaned dataset
df2 = pd.read_csv(clean_data_path)
print(df2.keys())
print(df2['Weather_conditions'].describe())
# Converting qualitative data column to quantitative|| Label Encoding
label_encoder = LabelEncoder()
df2['weather_conditions_encoded'] = label_encoder.fit_transform(df['Weather_conditions'])
# One-Hot Encoding
# df_encoded = pd.get_dummies(df, columns=['qualitative_column'], prefix=['qualitative_column'])
# working with text data
# Dealing with missing values
# Replacing with most frequent
most_frequent_categoryEd = df2['Educational_level'].mode()[0]
df2['Educational_level'].fillna(most_frequent_categoryEd, inplace=True)
most_frequent_categoryAAO = df2['Area_accident_occured'].mode()[0]
df2['Area_accident_occured'].fillna(most_frequent_categoryAAO, inplace=True)
most_frequent_categoryDV = df2['Defect_of_vehicle'].mode()[0]
df2['Defect_of_vehicle'].fillna(most_frequent_categoryDV, inplace=True)
most_frequent_categoryFC = df2['Fitness_of_casuality'].mode()[0]
df2['Fitness_of_casuality'].fillna(most_frequent_categoryFC, inplace=True)
most_frequent_categoryWC = df2['Work_of_casuality'].mode()[0]
df2['Work_of_casuality'].fillna(most_frequent_categoryWC, inplace=True)
most_frequent_categoryVM = df2['Vehicle_movement'].mode()[0]
df2['Vehicle_movement'].fillna(most_frequent_categoryVM, inplace=True)
most_frequent_categoryTC = df2['Type_of_collision'].mode()[0]
df2['Type_of_collision'].fillna(most_frequent_categoryTC, inplace=True)
most_frequent_categoryRST = df2['Road_surface_type'].mode()[0]
df2['Road_surface_type'].fillna(most_frequent_categoryRST, inplace=True)
most_frequent_categoryTJ = df2['Types_of_Junction'].mode()[0]
df2['Types_of_Junction'].fillna(most_frequent_categoryTJ, inplace=True)
most_frequent_categoryRA = df2['Road_allignment'].mode()[0]
df2['Road_allignment'].fillna(most_frequent_categoryRA, inplace=True)
most_frequent_categoryLM = df2['Lanes_or_Medians'].mode()[0]
df2['Lanes_or_Medians'].fillna(most_frequent_categoryLM, inplace=True)
most_frequent_categorySYV = df2['Service_year_of_vehicle'].mode()[0]
df2['Service_year_of_vehicle'].fillna(most_frequent_categorySYV, inplace=True)
most_frequent_categoryTV = df2['Type_of_vehicle'].mode()[0]
df2['Type_of_vehicle'].fillna(most_frequent_categoryTV, inplace=True)
most_frequent_categoryVDR = df2['Vehicle_driver_relation'].mode()[0]
df2['Vehicle_driver_relation'].fillna(most_frequent_categoryVDR, inplace=True)
most_frequent_categoryDE = df2['Driving_experience'].mode()[0]
df2['Driving_experience'].fillna(most_frequent_categoryDE, inplace=True)
most_frequent_categoryOV = df2['Owner_of_vehicle'].mode()[0]
df2['Owner_of_vehicle'].fillna(most_frequent_categoryOV, inplace=True)
# KNN like replacement
# df['text_column'].interpolate(method='linear', inplace=True)
print("ALL DATA IS CLEAN")
clean_data_pathA = 'D:\ClassSem5\Subjects\ML\Project files\datasets\Datasets\Cleaned_Stage2_RTA_Dataset.csv'
#df2.to_csv(clean_data_pathA, index=False)
#df3 = pd.read_csv(clean_data_pathA)
#print(df3.isnull().sum())
