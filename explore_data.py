import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the dataset (change to the correct filename if different)
df = pd.read_csv('Churn_Modelling.csv')

# Show the first five rows
print("First 5 rows:")
print(df.head())

# Data shape
print("Shape of data (rows, columns):", df.shape)

# Data types and null values
print("\nData Info:")
print(df.info())

# Basic statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Missing values per column
print("\nMissing Values per Column:")
print(df.isnull().sum())

df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male=1, Female=0 (usually)

df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

scaler = StandardScaler()
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

df[num_cols] = scaler.fit_transform(df[num_cols])

print(df.head())
