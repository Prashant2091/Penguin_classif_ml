import streamlit as st
import pandas as pd
import joblib
import numpy as np

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

# Main function to load the model and perform predictions
def main():
    st.title("Penguin Species Prediction App")
    st.sidebar.title("Choose Model")

    # Add a file uploader to let the user upload the model file
    uploaded_file = st.sidebar.file_uploader("Upload model file", type=["pkl"])
    if uploaded_file is not None:
        # Save the uploaded file
        with open("penguins_clf_fixed.pkl", "wb") as f:
            f.write(uploaded_file.getvalue())
        st.sidebar.write("Model uploaded successfully!")

    # Load the model if it exists
    model_path = "penguins_clf_fixed.pkl"  # Path to the saved model file
    if st.sidebar.button("Load Model"):
        try:
            # Load the model
            model = joblib.load(model_path)

            # Manually convert the dtype of the node array if necessary
            if isinstance(model, np.ndarray) and model.dtype.names != ('left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples', 'missing_go_to_left'):
                model = model.view(np.recarray)  # Convert to recarray
                model.dtype = [('left_child', '<i8'), ('right_child', '<i8'), ('feature', '<i8'), ('threshold', '<f8'), ('impurity', '<f8'), ('n_node_samples', '<i8'), ('weighted_n_node_samples', '<f8'), ('missing_go_to_left', 'u1')]

            st.write("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        # Streamlit app title and input fields
        st.title("Palmer Penguin Species Prediction")
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
            # Preprocess input data
            input_data = preprocess_input(bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex_female, sex_male, island_Biscoe, island_Dream, island_Torgersen)
            try:
                # Make prediction
                prediction = model.predict(input_data)
                # Display prediction
                st.write("Predicted Species:", prediction)
            except Exception as e:
                st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
