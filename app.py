import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Load and preprocess your data (reuse your preprocessing pipeline)
@st.cache_data
def load_data_and_model():
    df = pd.read_csv('Churn_Modelling.csv')
    df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

    # Encode Gender
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])

    # One-hot encode Geography
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Include all relevant features (including HasCrCard and IsActiveMember)
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model, scaler, X.columns.tolist()  # ✅ Return feature names to align input_df later


# Load model, scaler, and feature order
model, scaler, feature_order = load_data_and_model()

# 2. Streamlit UI
st.title("Bank Customer Churn Prediction")

# User inputs
gender = st.selectbox('Gender', ['Female', 'Male'])
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
credit_score = st.number_input('Credit Score', 300, 900, 650)
age = st.number_input('Age', 18, 100, 35)
tenure = st.number_input('Tenure (years)', 0, 10, 5)
balance = st.number_input('Balance', 0.0, 250000.0, 50000.0)
num_products = st.number_input('Num of Products', 1, 4, 1)
estimated_salary = st.number_input('Estimated Salary', 0.0, 200000.0, 50000.0)
has_cr_card = st.selectbox('Has Credit Card?', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member?', ['Yes', 'No'])

# Prepare input for model
gender_val = 1 if gender == 'Male' else 0
geo_germany = 1 if geography == 'Germany' else 0
geo_spain = 1 if geography == 'Spain' else 0
has_cr_card_val = 1 if has_cr_card == 'Yes' else 0
is_active_member_val = 1 if is_active_member == 'Yes' else 0

# Scale numeric inputs
input_data = np.array([[credit_score, age, tenure, balance, num_products, estimated_salary]])
input_data_scaled = scaler.transform(input_data)

# Build input dictionary
input_dict = {
    'CreditScore': input_data_scaled[0][0],
    'Age': input_data_scaled[0][1],
    'Tenure': input_data_scaled[0][2],
    'Balance': input_data_scaled[0][3],
    'NumOfProducts': input_data_scaled[0][4],
    'EstimatedSalary': input_data_scaled[0][5],
    'Gender': gender_val,
    'HasCrCard': has_cr_card_val,
    'IsActiveMember': is_active_member_val,
    'Geography_Germany': geo_germany,
    'Geography_Spain': geo_spain
}

input_df = pd.DataFrame([input_dict])

# ✅ Reorder columns to match training data
input_df = input_df[feature_order]

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn! (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer is NOT likely to churn. (Probability: {probability:.2f})")
