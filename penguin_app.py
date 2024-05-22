import streamlit as st
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

# Function to train and save the model
def train_and_save_model():
    # Load the dataset
    df = pd.read_csv("penguins.csv")

    # Preprocess the data
    # (Add your preprocessing code here)

    # Define features and target variable
    X = df.drop(columns=['species'])
    y = df['species']

    # Train a Decision Tree Classifier
    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    # Save the model
    joblib.dump(clf, "penguins_clf.pkl")
    st.write("Model trained and saved successfully!")

# Load the model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        st.write("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main function to load the model and perform predictions
def main():
    st.title("Penguin Species Prediction App")
    st.sidebar.title("Choose Model")

    # Add a file uploader to let the user upload the model file
    uploaded_file = st.sidebar.file_uploader("Upload model file", type=["pkl"])
    if uploaded_file is not None:
        # Save the uploaded file
        with open("uploaded_model.pkl", "wb") as f:
            f.write(uploaded_file.getvalue())
        st.sidebar.write("Model uploaded successfully!")

    # Load the model if it exists
    model_path = "uploaded_model.pkl"  # Path to the saved model file
    if st.sidebar.button("Load Model"):
        model = load_model(model_path)
        if model is None:
            st.warning("Please upload a valid model file.")
        else:
            st.success("Model loaded successfully!")

# Run the main function
if __name__ == "__main__":
    main()
