import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


# Load the MOdel
model=tf.keras.models.load_model("model.h5")

# Load the encoders and scaler
with open("encoder_gender.pkl", "rb") as f:
    label_encoder_gender=pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo=pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler=pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")

# User inputs
geography=st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender=st.selectbox("Gender", label_encoder_gender.classes_)
age=st.slider("Age", 18,92)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit Score")
estimated_salary=st.number_input("Estimeted Salary")
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider("Number of Products", 1,4)
has_credit_card=st.selectbox("Has Credit Card",[0,1])
is_active_user=st.selectbox("Is Active User", [0,1])

# Prepare the input data
input_data=pd.DataFrame({
    "CreditScore":[credit_score],
    "Geography":[geography],
    "Gender":[label_encoder_gender.transform([gender])[0]],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[num_of_products],
    "HasCrCard":[has_credit_card],
    "IsActiveMember":[is_active_user],
    "EstimatedSalary":[estimated_salary]
})


# one Hot Encoded "Geography"
encoder_geo= onehot_encoder_geo.transform([[geography]])
encoded_geo=pd.DataFrame(encoder_geo.toarray(), columns= onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data (drop Geography column first)
input_data=pd.concat([input_data.drop(['Geography'], axis=1).reset_index(drop=True), encoded_geo], axis=1)

# Scale the input data
input_data_scaled=scaler.transform(input_data)

# Predict Churn
Prediction=model.predict(input_data_scaled)
Prediction_prop=Prediction[0][0]

st.write(f"Churn Probability : {Prediction_prop:.2f}")
if Prediction_prop>0.5:
    st.write("The Customer is likely to churn")
else:
    st.write("The Customer is not likely to churn")