import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Diabetes Prediction System")

gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", 1, 100)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking = st.number_input("Smoking History (Encoded)", 0, 5)
bmi = st.number_input("BMI", 10.0, 60.0)
hba1c = st.number_input("HbA1c Level", 3.0, 15.0)
glucose = st.number_input("Blood Glucose Level", 50, 400)

gender = 1 if gender == "Male" else 0

if st.button("Predict"):
    data = np.array([[gender, age, hypertension, heart_disease,
                      smoking, bmi, hba1c, glucose]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Patient has Diabetes")
    else:
        st.success("Patient does NOT have Diabetes")