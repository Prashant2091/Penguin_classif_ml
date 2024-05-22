import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

df = load_data()

# Sidebar
st.sidebar.subheader("User Input Features")
selected_features = st.sidebar.selectbox("Features", df.columns[:-1])

# Load the model
model_path = "penguins_clf.pkl"
load_clf = joblib.load(model_path)

# Title of the app
st.title("Palmer Penguins Species Prediction")

# Show the dataset
if st.checkbox("Show DataFrame"):
    st.write(df)

# Input form to get user input
def user_input_features():
    input_features = []
    for feature in df.columns[:-1]:
        value = st.sidebar.slider(f"Select {feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
        input_features.append(value)
    return np.array(input_features).reshape(1, -1)

input_data = user_input_features()

# Make predictions
prediction = load_clf.predict(input_data)
prediction_proba = load_clf.predict_proba(input_data)

# Display predictions
species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.subheader("Prediction")
st.write(species[prediction][0])

st.subheader("Prediction Probability")
st.write(prediction_proba)
