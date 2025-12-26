import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the saved model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scale.pkl')

st.title("Titanic Survival Predictor")
st.write("Enter passenger details to see if they would have survived the shipwreck.")

# 2. Create the input form
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])
    sex = st.selectbox("Sex", ["Female", "Male"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
    fare = st.number_input("Fare Paid", min_value=0.0, value=32.0)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)
    embarked = st.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
    has_cabin = st.checkbox("Had a recorded Cabin?")

# 3. Pre-process the input exactly like the training data
# Convert Sex
sex_male = 1 if sex == "Male" else 0

# Convert Embarked (Dummy variables)
emb_q = 1 if "Q" in embarked else 0
emb_s = 1 if "S" in embarked else 0

# Create the feature array (Order must match your X training data!)
# [Pclass, Age, SibSp, Parch, Fare, Has_Cabin, Sex_male, Emb_Q, Emb_S]
input_data = np.array([[pclass, age, sibsp, parch, fare, int(has_cabin), sex_male, emb_q, emb_s]])

# 4. Scale and Predict
if st.button("Predict Survival"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    if prediction[0] == 1:
        st.success(f"The passenger likely SURVIVED (Confidence: {probability:.2%})")
    else:
        st.error(f"The passenger likely DIED (Confidence: {1-probability:.2%})")