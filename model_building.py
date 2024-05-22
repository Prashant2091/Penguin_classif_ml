# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 04:36:55 2022

@author: Satyam
"""

import pandas as pd
penguins = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/streamlit_freecodecamp/main/app_8_classification_penguins/penguins_cleaned.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = penguins.copy()
target = 'species'
encode = ['sex','island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)

# Separating X and y
X = df.drop('species', axis=1)
Y = df['species']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 04:36:55 2022

@author: Satyam
"""
#!pip install streamlit
import joblib
import pandas as pd
import streamlit as st
penguins = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/streamlit_freecodecamp/main/app_8_classification_penguins/penguins_cleaned.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = penguins.copy()
target = 'species'
encode = ['sex','island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)

# Separating X and y
X = df.drop('species', axis=1)
Y = df['species']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
# Load the model file and inspect its contents
import streamlit as st
import numpy as np
import joblib

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

# If the loaded model is a NumPy array, print its shape and dtype
if isinstance(model, np.ndarray):
    st.write("Array shape:", model.shape)
    st.write("Array dtype:", model.dtype)
    st.write("Array contents:")
    for item in model:
        st.write(item)

# If the loaded model is a scikit-learn model, print its attributes
elif hasattr(model, 'estimators_'):
    st.write("Model attributes:", model.__dict__.keys())

# If the loaded model is neither a NumPy array nor a scikit-learn model, display an error
else:
    st.error("Loaded model type not recognized. Please ensure that the correct model file is used.")

# Additional Streamlit app code (e.g., user input features, prediction logic) goes here

