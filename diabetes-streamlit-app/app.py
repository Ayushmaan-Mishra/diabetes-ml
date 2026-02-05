import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

# Load dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Split data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardization
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

# Train model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Enter patient details below:")

pregnancies = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
blood_pressure = st.number_input("Blood Pressure")
skin_thickness = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):
    input_data = np.array([
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]).reshape(1, -1)

    std_data = scaler.transform(input_data)
    prediction = classifier.predict(std_data)

    if prediction[0] == 0:
        st.success("The person is NOT diabetic")
    else:
        st.error("The person IS diabetic")
