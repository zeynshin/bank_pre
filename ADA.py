import streamlit as st # web-based user interface
import joblib # load saved model files
import pandas as pd # data handling
import numpy as np # numerical operations

# configure the web page appearance
st.set_page_config(page_title="Bank Predictor", layout="centered")

# cache the model to avoid reloading on every user interaction
@st.cache_resource
def load_bundle():
    try:
        # load the trained model/pipeline bundle
        bundle = joblib.load("model_bundle.pkl")
        return bundle
    except FileNotFoundError:
        # show error message if the file is missing
        st.error("Error: 'model_bundle.pkl' file not found!")
        return None

bundle = load_bundle()

if bundle:
    # extract the model from the bundle (checks if it's stored in a dictionary)
    model = bundle["model"] if isinstance(bundle, dict) else bundle

    # app titles and descriptions
    st.title("Bank Marketing Prediction")
    st.markdown("Enter customer information to predict the campaign outcome.")

    # visual layout: split input fields into two columns
    col1, col2 = st.columns(2)

    with col1:
        # numerical and categorical inputs for the first column
        age = st.slider("Age", 18, 100, 30)
        job = st.selectbox("Job", ["admin.", "technician", "services", "management", "blue-collar", "retired", "entrepreneur", "housemaid", "self-employed", "student", "unemployed"])
        marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
        education = st.selectbox("Education", ["primary", "secondary", "tertiary"])
        balance = st.number_input("Balance", value=0)
        housing = st.selectbox("Housing Loan", ["yes", "no"])
        loan = st.selectbox("Personal Loan", ["yes", "no"])

    with col2:
        # inputs for the second column
        contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"])
        day = st.number_input("Day", min_value=1, max_value=31, value=15)
        month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
        duration = st.number_input("Last Call Duration (sec)", value=100)
        campaign = st.number_input("Campaign Count", value=1)
        pdays = st.number_input("Pdays (-1 if never)", value=-1)
        previous = st.number_input("Previous Contacts", value=0)
        poutcome = st.selectbox("Poutcome", ["unknown", "success", "failure", "other"])

    st.divider() # visual separator

    # trigger prediction when button is clicked
    if st.button("Predict", use_container_width=True):

        # convert user inputs into a dataframe with correct column names
        input_df = pd.DataFrame([{
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": "no", # default value for missing field
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

        try:
            # perform prediction and calculate probability
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            # display results based on the prediction outcome
            if prediction == 1:
                st.success(f"Result: **SUBSCRIBES** (Probability: %{prob*100:.2f})")
            else:
                st.warning(f"Result: **DOES NOT SUBSCRIBE** (Probability: %{prob*100:.2f})")
                
        except Exception as e:
            # handle errors like mismatched feature names or types
            st.error(f"An error occurred during prediction. There might be a column mismatch: {e}")