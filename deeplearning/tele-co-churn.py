import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

df = pd.read_csv("Telco-Customer-Churn.csv")


print ("Dataset Info\n")
print(df.info())
print("\n Class Distribution: \n")
print(df['Churn'].value_counts())
print("\nSample Data:\n", df.head())


# handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')
df.fillna({'TotalCharges' : df['TotalCharges'].median()}, inplace = True)

#Encode Categorical Variables
label_encoder = LabelEncoder()
for column in df.select_dtypes(include = ['object']).columns:
    if column != 'Churn':
        df[column] = label_encoder.fit_transform(df[column])
        
# Encode target variable 
df['Churn'] = label_encoder.fit_transform(df['Churn'])

# Encode target variable 
df['Churn'] = label_encoder.fit_transform(df['Churn'])

#scale numerical features
scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

#feature and target
X = df.drop(columns = ['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

#Train Initial Model
rf_model = RandomForestClassifier(random_state = 42)
rf_model.fit(X_train, y_train)

#evaluate initial model
y_pred = rf_model.predict(X_test)
accuracy_initial = accuracy_score(y_test, y_pred)




 




