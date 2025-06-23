import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Import necessary classes used in the pipeline
# These imports are crucial for joblib.load to work correctly in the Streamlit environment
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier # Import the actual model class used in the pipeline
from sklearn.linear_model import LogisticRegression # Include if Logistic Regression was also part of the pipeline (e.g., in preprocessor steps)

# Define the expected column names after preprocessing
# This list must exactly match the columns and their order that the preprocessor outputs
# and that the trained model expects.
# Based on the notebook's df_processed_transformed and X_train_res columns:
processed_feature_names = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
                           'gender_Female', 'gender_Male', 'gender_Other', 'ever_married_No',
                           'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked',
                           'work_type_Private', 'work_type_Self-employed', 'work_type_children',
                           'Residence_type_Rural', 'Residence_type_Urban', 'smoking_status_Unknown',
                           'smoking_status_formerly smoked', 'smoking_status_never smoked',
                           'smoking_status_smokes']


# Load the full pipeline
# This file must be in the same directory as app.py when deployed
try:
    # Ensure the filename matches the saved file name
    full_pipeline = joblib.load('full_pipeline.pkl')
    st.success("Full pipeline loaded successfully!")
except FileNotFoundError:
    st.error("Error: Pipeline file not found. Make sure 'full_pipeline.pkl' is in the same directory as app.py.")
    st.stop() # Stop execution if files are not found
except Exception as e:
    st.error(f"Error loading pipeline: {e}")
    st.stop() # Stop execution if other loading errors occur


st.title("Aplikasi Deteksi Penyakit Stroke")

st.write("Aplikasi ini menggunakan model Machine Learning untuk memprediksi risiko stroke berdasarkan input pengguna.")

# Create input fields for the user, matching the original dataset columns
st.header("Input Data Pasien:")

# Collect user input for each original feature
# Ensure data types and min/max values are appropriate
gender = st.selectbox("Jenis Kelamin", ['Female', 'Male', 'Other'])
age = st.number_input("Usia", min_value=0.0, max_value=120.0, value=30.0, help="Usia pasien")
hypertension = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: 'Ya' if x == 1 else 'Tidak', help="Riwayat Hipertensi (0: Tidak, 1: Ya)")
heart_disease = st.selectbox("Penyakit Jantung", [0, 1], format_func=lambda x: 'Ya' if x == 1 else 'Tidak', help="Riwayat Penyakit Jantung (0: Tidak, 1: Ya)")
ever_married = st.selectbox("Pernah Menikah", ['Yes', 'No'], help="Status pernikahan")
work_type = st.selectbox("Tipe Pekerjaan", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], help="Jenis pekerjaan")
Residence_type = st.selectbox("Tipe Tempat Tinggal", ['Urban', 'Rural'], help="Tipe tempat tinggal")
avg_glucose_level = st.number_input("Rata-rata Tingkat Glukosa", min_value=0.0, value=100.0, help="Rata-rata tingkat glukosa dalam darah")
bmi = st.number_input("BMI", min_value=0.0, value=25.0, help="Indeks Massa Tubuh (BMI)") # Imputation was used for missing BMI
smoking_status = st.selectbox("Status Merokok", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'], help="Status merokok")

# Create a dictionary from user input
user_input = {
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

# Convert user input to a pandas DataFrame
user_input_df = pd.DataFrame([user_input])

# Ensure the order of columns in the input DataFrame matches the order the preprocessor expects
# This order should be the same as the original columns (excluding 'id' and 'stroke')
original_columns_order = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
user_input_df = user_input_df[original_columns_order]


# Add a button to make a prediction
if st.button("Prediksi Risiko Stroke"):
    try:
        # Use the full pipeline to preprocess and predict in one step
        # The pipeline expects the original features as input
        prediction = full_pipeline.predict(user_input_df)
        prediction_proba = full_pipeline.predict_proba(user_input_df)[:, 1] # Probability of stroke (class 1)

        st.subheader("Hasil Prediksi:")
        if prediction[0] == 1:
            st.write("Hasil: **Risiko Stroke Tinggi**")
            st.warning(f"Probabilitas Stroke: {prediction_proba[0]:.4f}")
        else:
            st.write("Hasil: **Risiko Stroke Rendah**")
            st.success(f"Probabilitas Stroke: {prediction_proba[0]:.4f}")

        st.info("Catatan: Ini adalah prediksi berdasarkan model machine learning dan tidak menggantikan diagnosis medis profesional.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        # Optionally, print traceback for debugging
        # import traceback
        # st.error(traceback.format_exc())
