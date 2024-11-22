import pickle
import streamlit as st
import numpy as np
import sklearn

model_logistic = pickle.load(open('model_diabetes.sav', 'rb'))
model_random_forest = pickle.load(open('model_diabetes_random_forest.sav', 'rb'))
model_gaussian = pickle.load(open('model_diabetes_gaussian.sav', 'rb'))

st.markdown("<h1 style='text-align: center; color: darkblue;'>Diabetes Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #333399;'>Please Enter the Required Information:</h3>", unsafe_allow_html=True)


col1 = st.container()
with col1:
    model_choice = st.selectbox('Select the Model', ['Logistic Regression', 'Random Forest', 'Gaussian'])

    
    Pregnancies = st.number_input('Enter the Pregnancies value', key='Pregnancies')
    Glucose = st.number_input('Enter the Glucose value', key='Glucose')
    BloodPressure = st.number_input('Enter the Blood Pressure value', key='BloodPressure')
    SkinThickness = st.number_input('Enter the Skin Thickness value', key='SkinThickness')
    Insulin = st.number_input('Enter the Insulin value', key='Insulin')
    BMI = st.number_input('Enter the BMI value', key='BMI')
    DiabetesPedigreeFunction = st.number_input('Enter the Diabetes Pedigree Function value', key='DiabetesPedigreeFunction')
    Age = st.number_input('Enter the Age value', key='Age')


diabetes_diagnosis = ''

if st.button('Diabetes Prediction Test', key='predict'):
   
    if model_choice == 'Logistic Regression':
        model = model_logistic
    elif model_choice == 'Random Forest':
        model = model_random_forest
    else:
        model = model_gaussian

    
    diabetes_prediction = model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    
    if diabetes_prediction[0] == 1:
        diabetes_diagnosis = "<p style='color: red; font-size: 20px;'>The patient has diabetes</p>"
    else:
        diabetes_diagnosis = "<p style='color: green; font-size: 20px;'>The patient does not have diabetes</p>"

st.markdown(diabetes_diagnosis, unsafe_allow_html=True)
