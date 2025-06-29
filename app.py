import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('hospital_readmission_model.pkl')

st.title('Hospital Readmission Prediction')

# Input widgets
st.sidebar.header('Patient Information')
age = st.sidebar.slider('Age', 18, 100, 50)
gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'])
admission_type = st.sidebar.selectbox('Admission Type', ['Emergency', 'Urgent', 'Elective'])
diagnosis = st.sidebar.selectbox('Diagnosis', ['Heart Disease', 'Diabetes', 'Injury', 'Infection'])
num_lab = st.sidebar.slider('Lab Procedures', 1, 100, 50)
num_meds = st.sidebar.slider('Medications', 1, 35, 10)
num_out = st.sidebar.slider('Outpatient Visits', 0, 4, 2)
num_in = st.sidebar.slider('Inpatient Visits', 0, 4, 1)
num_emergency = st.sidebar.slider('Emergency Visits', 0, 4, 1)
num_diag = st.sidebar.slider('Diagnoses', 1, 9, 3)
a1c = st.sidebar.selectbox('A1C Result', ['Normal', 'Abnormal', 'Unknown'])

# Create input DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Admission_Type': [admission_type],
    'Diagnosis': [diagnosis],
    'Num_Lab_Procedures': [num_lab],
    'Num_Medications': [num_meds],
    'Num_Outpatient_Visits': [num_out],
    'Num_Inpatient_Visits': [num_in],
    'Num_Emergency_Visits': [num_emergency],
    'Num_Diagnoses': [num_diag],
    'A1C_Result': [a1c]
})

# Display input data
st.subheader('Patient Data')
st.write(input_data)

# Predict and display results
if st.button('Predict Readmission Risk'):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    
    risk_level = "High Risk" if prediction == 1 else "Low Risk"
    risk_prob = proba[1] * 100  # Probability of readmission
    
    st.subheader('Prediction Result')
    st.success(f'Readmission Risk: **{risk_level}**')
    st.info(f'Probability of Readmission: **{risk_prob:.1f}%**')
    
    # Visual indicator
    st.progress(int(risk_prob))
    if risk_prob > 70:
        st.warning('⚠️ High probability of readmission - Consider intervention')
    elif risk_prob > 40:
        st.info('🟠 Moderate readmission risk - Monitor closely')
    else:
        st.success('✅ Low readmission risk')

# Add footer
st.markdown("---")
st.caption("ML Model for Hospital Readmission Prediction | Built with Streamlit")
