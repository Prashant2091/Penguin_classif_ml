import streamlit as st
import pandas as pd
import joblib

# Define a function to load the model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        st.write("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess input data
def preprocess_input(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex_female, sex_male, island_Biscoe, island_Dream, island_Torgersen):
    # Create a DataFrame with the input values
    data = {
        'bill_length_mm': [bill_length_mm],
        'bill_depth_mm': [bill_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex_female': [sex_female],
        'sex_male': [sex_male],
        'island_Biscoe': [island_Biscoe],
        'island_Dream': [island_Dream],
        'island_Torgersen': [island_Torgersen]
    }
    input_df = pd.DataFrame(data)
    return input_df

def main():
    st.title("Palmer Penguin Species Prediction")
    
    # Load the model
    model_path = "penguins_clf.pkl"
    model = load_model(model_path)
    if model is None:
        st.error("Failed to load the model. Please upload a valid model file.")

    # Streamlit app input fields
    bill_length_mm = st.number_input("Bill Length (mm)", min_value=0.0, max_value=100.0, value=40.0)
    bill_depth_mm = st.number_input("Bill Depth (mm)", min_value=0.0, max_value=100.0, value=20.0)
    flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=0.0, max_value=100.0, value=200.0)
    body_mass_g = st.number_input("Body Mass (g)", min_value=0.0, max_value=10000.0, value=4000.0)
    sex_female = st.checkbox("Female")
    sex_male = st.checkbox("Male")
    island_Biscoe = st.checkbox("Biscoe")
    island_Dream = st.checkbox("Dream")
    island_Torgersen = st.checkbox("Torgersen")

    # Predict button
    if st.button("Predict"):
        if model is not None:
            # Preprocess input data
            input_data = preprocess_input(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex_female, sex_male, island_Biscoe, island_Dream, island_Torgersen)
            # Make prediction
            prediction = model.predict(input_data)
            # Display prediction
            st.write("Predicted Species:", prediction)

if __name__ == "__main__":
    main()
