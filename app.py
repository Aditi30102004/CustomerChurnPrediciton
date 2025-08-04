import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib # Used for saving/loading model and scaler

# --- DATA LOADING AND MODEL TRAINING (Optimized) ---

# Use @st.cache_resource to load the model and scaler only once
@st.cache_resource
def load_model_and_scaler():
    """
    Loads the pre-trained model and scaler. If they don't exist,
    it trains them and saves them to files for future use.
    """
    try:
        model = joblib.load('churn_model.joblib')
        scaler = joblib.load('scaler.joblib')
    except FileNotFoundError:
        # Load the dataset
        df = pd.read_csv('Churn_Modelling.csv')
        
        # Preprocessing
        df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])
        df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
        
        # Define features and target
        X = df.drop(columns=['Exited'])
        y = df['Exited']
        
        # Scale numerical features
        scaler = StandardScaler()
        num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        X[num_cols] = scaler.fit_transform(X[num_cols])
        
        # Train the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        # Save the model and scaler for future runs
        joblib.dump(model, 'churn_model.joblib')
        joblib.dump(scaler, 'scaler.joblib')
        
    return model, scaler

# Load the raw data for visualizations
@st.cache_data
def load_raw_data():
    df = pd.read_csv('Churn_Modelling.csv')
    return df

model, scaler = load_model_and_scaler()
raw_df = load_raw_data()

# --- SIDEBAR - USER INPUTS ---

st.sidebar.header("Customer Details")

# Create input fields in the sidebar
credit_score = st.sidebar.slider('Credit Score', 300, 850, 650)
geography = st.sidebar.selectbox('Geography', ('France', 'Germany', 'Spain'))
gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
age = st.sidebar.slider('Age', 18, 100, 35)
tenure = st.sidebar.slider('Tenure (years)', 0, 10, 5)
balance = st.sidebar.slider('Balance', 0.0, 250000.0, 0.0)
num_products = st.sidebar.selectbox('Number of Products', (1, 2, 3, 4))
has_cr_card = st.sidebar.selectbox('Has Credit Card?', ('Yes', 'No'))
is_active_member = st.sidebar.selectbox('Is Active Member?', ('Yes', 'No'))
estimated_salary = st.sidebar.slider('Estimated Salary', 0.0, 200000.0, 50000.0)

# --- MAIN PAGE ---

st.title("Customer Churn Prediction")
st.write("This app predicts whether a bank customer will churn or not based on their details. Use the sidebar to enter customer information.")

# --- PREDICTION LOGIC ---

# Prepare input for the model
gender_val = 1 if gender == 'Male' else 0
geo_germany = 1 if geography == 'Germany' else 0
geo_spain = 1 if geography == 'Spain' else 0
has_cr_card_val = 1 if has_cr_card == 'Yes' else 0
is_active_member_val = 1 if is_active_member == 'Yes' else 0

# Scale the numerical features
input_data_scaled = scaler.transform([[credit_score, age, tenure, balance, num_products, estimated_salary]])

# Create the final input dictionary in the correct order
input_dict = {
    'CreditScore': input_data_scaled[0][0],
    'Age': input_data_scaled[0][1],
    'Tenure': tenure,
    'Balance': input_data_scaled[0][3],
    'NumOfProducts': num_products,
    'EstimatedSalary': input_data_scaled[0][5],
    'Gender': gender_val,
    'HasCrCard': has_cr_card_val,
    'IsActiveMember': is_active_member_val,
    'Geography_Germany': geo_germany,
    'Geography_Spain': geo_spain
}

# The model expects a DataFrame with columns in a specific order.
# We get this order from the saved model.
expected_order = model.feature_names_in_
input_df = pd.DataFrame([input_dict])[expected_order]


# Display the prediction
st.header("Prediction Result")
if st.sidebar.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] # Probability of churn
    
    if prediction == 1:
        st.error(f"This customer is LIKELY to churn (Probability: {probability:.2f})")
    else:
        st.success(f"This customer is NOT likely to churn (Probability of churn: {1-probability:.2f})")

# --- DATA VISUALIZATIONS ---
st.markdown("---")
st.header("Dashboard: Churn Insights")

# Create two columns for charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Rate by Geography")
    geo_churn = raw_df.groupby('Geography')['Exited'].mean()
    st.bar_chart(geo_churn)

with col2:
    st.subheader("Churn Rate by Gender")
    gender_churn = raw_df.groupby('Gender')['Exited'].mean()
    st.bar_chart(gender_churn)

st.subheader("Overall Churn Distribution")
churn_counts = raw_df['Exited'].value_counts().rename(index={0: 'Not Churned', 1: 'Churned'})
st.bar_chart(churn_counts)
