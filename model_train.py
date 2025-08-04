import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# 1. Load the data
df = pd.read_csv('Churn_Modelling.csv')

# 2. Drop unnecessary columns
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# 3. Encode categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])                  # Gender: Female=0, Male=1
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# 4. Scale numerical features
scaler = StandardScaler()
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
df[num_cols] = scaler.fit_transform(df[num_cols])

# 5. Define target and features
X = df.drop(columns=['Exited'])
y = df['Exited']

# 6. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train models
logreg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(random_state=42)
logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)

# 8. Make predictions
logreg_preds = logreg.predict(X_test)
rf_preds = rf.predict(X_test)

# 9. Evaluate models
print("Logistic Regression:")
print("  Accuracy:", accuracy_score(y_test, logreg_preds))
print("  Precision:", precision_score(y_test, logreg_preds))
print("  Recall:", recall_score(y_test, logreg_preds))
print("  ROC-AUC:", roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1]))

print("\nRandom Forest:")
print("  Accuracy:", accuracy_score(y_test, rf_preds))
print("  Precision:", precision_score(y_test, rf_preds))
print("  Recall:", recall_score(y_test, rf_preds))
print("  ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
