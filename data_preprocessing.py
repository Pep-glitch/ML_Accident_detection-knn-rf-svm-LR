from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data_path = 'D:\ClassSem5\Subjects\ML\Project files\datasets\Datasets\Cleaned_Stage2_RTA_Dataset.csv'
data = pd.read_csv(data_path)
print(data['Accident_severity'].describe())
print(data['Accident_severity'].value_counts())
print("Road Surface Type")
print(data['Road_surface_type'].value_counts())
print("Road Surface Conditions")
print(data['Road_surface_conditions'].value_counts())
print("Light Conditions")
print(data['Light_conditions'].value_counts())

# Label Encoding
label_encoder = LabelEncoder()
data['Age_band_of_driver_encoded'] = label_encoder.fit_transform(data['Age_band_of_driver'])
data['Day_of_week_encoded'] = label_encoder.fit_transform(data['Day_of_week'])
data['Sex_of_driver_encoded'] = label_encoder.fit_transform(data['Sex_of_driver'])
data['Educational_level_encoded'] = label_encoder.fit_transform(data['Educational_level'])
data['Vehicle_driver_relation_encoded'] = label_encoder.fit_transform(data['Vehicle_driver_relation'])
data['Driving_experience_encoded'] = label_encoder.fit_transform(data['Driving_experience'])
data['Type_of_vehicle_encoded'] = label_encoder.fit_transform(data['Type_of_vehicle'])
data['Owner_of_vehicle_encoded'] = label_encoder.fit_transform(data['Owner_of_vehicle'])
data['Service_year_of_vehicle_encoded'] = label_encoder.fit_transform(data['Service_year_of_vehicle'])
data['Defect_of_vehicle_encoded'] = label_encoder.fit_transform(data['Defect_of_vehicle'])
data['Area_accident_occured_encoded'] = label_encoder.fit_transform(data['Area_accident_occured'])
data['Lanes_or_Medians_encoded'] = label_encoder.fit_transform(data['Lanes_or_Medians'])
data['Road_allignment_encoded'] = label_encoder.fit_transform(data['Road_allignment'])
data['Types_of_Junction_encoded'] = label_encoder.fit_transform(data['Types_of_Junction'])
data['Road_surface_type_encoded'] = label_encoder.fit_transform(data['Road_surface_type'])
data['Road_surface_conditions_encoded'] = label_encoder.fit_transform(data['Road_surface_conditions'])
data['Light_conditions_encoded'] = label_encoder.fit_transform(data['Light_conditions'])
data['Type_of_collision_encoded'] = label_encoder.fit_transform(data['Type_of_collision'])
data['Vehicle_movement_encoded'] = label_encoder.fit_transform(data['Vehicle_movement'])
data['Casualty_class_encoded'] = label_encoder.fit_transform(data['Casualty_class'])
data['Sex_of_casualty_encoded'] = label_encoder.fit_transform(data['Sex_of_casualty'])
data['Age_band_of_casualty_encoded'] = label_encoder.fit_transform(data['Age_band_of_casualty'])
data['Casualty_severity_encoded'] = label_encoder.fit_transform(data['Casualty_severity'])
data['Work_of_casuality_encoded'] = label_encoder.fit_transform(data['Work_of_casuality'])
data['Fitness_of_casuality_encoded'] = label_encoder.fit_transform(data['Fitness_of_casuality'])
data['Pedestrian_movement_encoded'] = label_encoder.fit_transform(data['Pedestrian_movement'])
data['Cause_of_accident_encoded'] = label_encoder.fit_transform(data['Cause_of_accident'])
data['Accident_severity_encoded'] = label_encoder.fit_transform(data['Accident_severity'])


new_data_path = 'D:\ClassSem5\Subjects\ML\Project files\datasets\Datasets\Encoded_RTA_Dataset.csv'
#data.to_csv(new_data_path, index=False)
print("Data Encoded Successfully")

data2 = pd.read_csv(new_data_path)
drop_columns = ['Time', 'Age_band_of_driver','Day_of_week','Sex_of_driver','Educational_level','Vehicle_driver_relation','Driving_experience', 'Type_of_vehicle','Owner_of_vehicle','Service_year_of_vehicle','Defect_of_vehicle', 'Area_accident_occured', 'Lanes_or_Medians', 'Road_allignment', 'Types_of_Junction', 'Road_surface_type', 'Road_surface_conditions', 'Light_conditions',  'Type_of_collision', 'Vehicle_movement', 'Casualty_class', 'Sex_of_casualty', 'Age_band_of_casualty','Casualty_severity', 'Work_of_casuality', 'Fitness_of_casuality',  'Pedestrian_movement', 'Cause_of_accident', 'Accident_severity' ,'Weather_conditions']
data2.drop(columns=drop_columns, inplace=True)
latest_path = 'D:\ClassSem5\Subjects\ML\Project files\datasets\Datasets\Clean_Encoded_RTA_Dataset.csv'
data2.to_csv(latest_path, index=False)
print("Columns Removed")

