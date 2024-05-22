import streamlit as st
import joblib
import pandas as pd

# Load the model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    return joblib.load(model_path)

# Load the model
model_path = "penguins_clf.pkl"
load_clf = load_model(model_path)

# Define the input fields
st.title("Palmer Penguin Species Prediction")
bill_length_mm = st.number_input("Bill Length (mm)", min_value=0.0, max_value=100.0, value=40.0)
bill_depth_mm = st.number_input("Bill Depth (mm)", min_value=0.0, max_value=100.0, value=20.0)
flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=0.0, max_value=100.0, value=200.0)
body_mass_g = st.number_input("Body Mass (g)", min_value=0.0, max_value=10000.0, value=4000.0)
sex_female = st.checkbox("Female")
sex_male = st.checkbox("Male")
island_biscoe = st.checkbox("Biscoe")
island_dream = st.checkbox("Dream")
island_torgersen = st.checkbox("Torgersen")

# Prepare input data
data = {
    'bill_length_mm': bill_length_mm,
    'bill_depth_mm': bill_depth_mm,
    'flipper_length_mm': flipper_length_mm,
    'body_mass_g': body_mass_g,
    'sex_female': sex_female,
    'sex_male': sex_male,
    'island_Biscoe': island_biscoe,
    'island_Dream': island_dream,
    'island_Torgersen': island_torgersen
}

input_df = pd.DataFrame([data])

# Make prediction
prediction = load_clf.predict(input_df)

# Display prediction
st.subheader("Prediction")
st.write(prediction)
