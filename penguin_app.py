import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species!

Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]  # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)

# Debugging: Print the current working directory
st.write(f"Current working directory: {os.getcwd()}")

# Debugging: Print the contents of the current working directory
st.write("Contents of the current working directory:")
st.write(os.listdir(os.getcwd()))

# Reads in saved classification model
import streamlit as st
import joblib
import numpy as np

st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species!

Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

# Load the trained model
model_path = 'penguins_clf.pkl'
try:
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
except FileNotFoundError:
    st.error(f"Model file {model_path} not found. Please ensure the file is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Inspect the loaded model
st.write("Model type:", type(model))

# If the loaded model is a scikit-learn model, print its attributes
if hasattr(model, 'tree_'):
    st.write("Model attributes:", model.__dict__.keys())

# If the loaded model is a NumPy array, print its shape, dtype, and contents
elif isinstance(model, np.ndarray):
    st.write("Array shape:", model.shape)
    st.write("Array dtype:", model.dtype)
    st.write("Array contents:")
    for item in model:
        st.write(item)

# If the loaded model is neither a scikit-learn model nor a NumPy array, display an error
else:
    st.error("Loaded model type not recognized. Please ensure that the correct model file is used.")

# Additional Streamlit app code (e.g., user input features, prediction logic) goes here

# Apply model to make predictions
if 'load_clf' in locals():
    if isinstance(load_clf, RandomForestClassifier):  # Ensure it's the expected model type
        prediction = load_clf.predict(df)
        prediction_proba = load_clf.predict_proba(df)

        st.subheader('Prediction')
        penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
        st.write(penguins_species[prediction])

        st.subheader('Prediction Probability')
        st.write(prediction_proba)
    else:
        st.error("The loaded model is not a RandomForestClassifier instance.")
