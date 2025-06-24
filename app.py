
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

# Set the title of the app
st.title('Stroke Prediction App')

# Add input fields for each feature
st.header('Enter Patient Data:')

# Define input fields based on the features used in the model
# Numerical features: 'age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease'

# Default values that are likely to result in a high stroke risk prediction
default_data = {
    'gender': ['Male'],
    'age': [70.0],
    'hypertension': [1],
    'heart_disease': [1],
    'ever_married': ['Yes'],
    'work_type': ['Private'],
    'Residence_type': ['Urban'],
    'avg_glucose_level': [200.0],
    'bmi': [35.0],
    'smoking_status': ['formerly smoked']
}

# Convert default data to DataFrame
default_df = pd.DataFrame(default_data) # No need for list around default_data for single row

age = st.number_input('Age', min_value=0.0, max_value=120.0, value=float(default_df['age'].iloc[0]), step=0.1)
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, value=float(default_df['avg_glucose_level'].iloc[0]), step=0.1)
bmi = st.number_input('BMI', min_value=0.0, value=float(default_df['bmi'].iloc[0]), step=0.1)
hypertension = st.selectbox('Hypertension', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', index=int(default_df['hypertension'].iloc[0]))
heart_disease = st.selectbox('Heart Disease', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', index=int(default_df['heart_disease'].iloc[0]))

# Categorical features: 'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'
gender_options = ['Female', 'Male', 'Other']
gender = st.selectbox('Gender', options=gender_options, index=gender_options.index(default_df['gender'].iloc[0]))

ever_married_options = ['Yes', 'No']
ever_married = st.selectbox('Ever Married', options=ever_married_options, index=ever_married_options.index(default_df['ever_married'].iloc[0]))

work_type_options = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
work_type = st.selectbox('Work Type', options=work_type_options, index=work_type_options.index(default_df['work_type'].iloc[0]))

Residence_type_options = ['Urban', 'Rural']
Residence_type = st.selectbox('Residence Type', options=Residence_type_options, index=Residence_type_options.index(default_df['Residence_type'].iloc[0]))

smoking_status_options = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']
smoking_status = st.selectbox('Smoking Status', options=smoking_status_options, index=smoking_status_options.index(default_df['smoking_status'].iloc[0]))

# Create a button to trigger prediction
if st.button('Predict Stroke'):
    # Create a dictionary from user input
    user_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }

    # Convert user data to DataFrame
    user_df = pd.DataFrame([user_data])

    # Load the trained model and preprocessor
    try:
        # Use the path to the balanced model
        model_path = 'best_xgb_model_balanced.pkl'
        preprocessor_path = 'preprocessor.pkl'

        if not os.path.exists(model_path):
             st.error(f"Error: Model file not found at {model_path}. Please ensure it's in the correct directory.")
        elif not os.path.exists(preprocessor_path):
             st.error(f"Error: Preprocessor file not found at {preprocessor_path}. Please ensure it's in the correct directory.")
        else:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)

            # Preprocess the user input
            user_processed = preprocessor.transform(user_df)

            # Make prediction
            prediction = model.predict(user_processed)
            prediction_proba = model.predict_proba(user_processed)[:, 1] # Probability of stroke

            # Display the prediction result
            st.header('Prediction Result:')
            if prediction[0] == 1:
                st.error(f'Based on the input data, there is a high risk of stroke. (Probability: {prediction_proba[0]:.4f})')
            else:
                st.success(f'Based on the input data, the risk of stroke is low. (Probability: {prediction_proba[0]:.4f})')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

