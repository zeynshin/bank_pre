import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Bank Predictor", layout="centered")

# Load the model (assuming it's a Pipeline object)
@st.cache_resource
def load_bundle():
    try:
        # When the pipeline is loaded, 'preprocessor' steps run automatically
        bundle = joblib.load("model_bundle.pkl")
        return bundle
    except FileNotFoundError:
        st.error("Error: 'model_bundle.pkl' file not found!")
        return None

bundle = load_bundle()

if bundle:
    # Extract based on bundle structure (if only model was saved, use bundle['model'])
    model = bundle["model"] if isinstance(bundle, dict) else bundle

    st.title("Bank Marketing Prediction")
    st.markdown("Enter customer information to predict the campaign outcome.")

    # Split inputs into two columns (looks cleaner)
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 30)
        job = st.selectbox("Job", ["admin.", "technician", "services", "management", "blue-collar", "retired", "entrepreneur", "housemaid", "self-employed", "student", "unemployed"])
        marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
        education = st.selectbox("Education", ["primary", "secondary", "tertiary"])
        balance = st.number_input("Balance", value=0)
        housing = st.selectbox("Housing Loan", ["yes", "no"])
        loan = st.selectbox("Personal Loan", ["yes", "no"])

    with col2:
        contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"])
        day = st.number_input("Day", min_value=1, max_value=31, value=15)
        month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
        duration = st.number_input("Last Call Duration (sec)", value=100)
        campaign = st.number_input("Campaign Count", value=1)
        pdays = st.number_input("Pdays (-1 if never)", value=-1)
        previous = st.number_input("Previous Contacts", value=0)
        poutcome = st.selectbox("Poutcome", ["unknown", "success", "failure", "other"])

    st.divider()

    if st.button("Predict", use_container_width=True):
        # Column names and format used during training
        # IMPORTANT: Column names must match the training DataFrame
        input_df = pd.DataFrame([{
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": "no", # Added as default
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }])

        # Prediction process
        try:
            # If 'bundle' is a Pipeline, no need for reindex; it handles preprocessing
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            # Result Screen
            if prediction == 1:
                st.success(f"Result: **SUBSCRIBES** (Probability: %{prob*100:.2f})")
            else:
                st.warning(f"Result: **DOES NOT SUBSCRIBE** (Probability: %{prob*100:.2f})")
                
        except Exception as e:
            st.error(f"An error occurred during prediction. There might be a column mismatch: {e}")