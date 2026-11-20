import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction", page_icon="📊", layout="wide"
)


# Application Header
st.title("📊 Customer Churn Prediction")

st.markdown(
    """
    Predict customer churn using Machine Learning.

    Enter customer details below to estimate their likelihood
    of leaving the telecom company.
    """
)

st.divider()

# Sidebar
st.sidebar.title("About This Project")

st.sidebar.info(
    """
    **Customer Churn Prediction**

    This application uses Machine Learning
    to predict whether a telecom customer
    is likely to leave the company.

    **Technology Used:**
    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    """
)

st.sidebar.caption("Machine Learning Project")


# Get the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model file path
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")

# Load the trained model
model = joblib.load(MODEL_PATH)

st.success("Machine learning model loaded successfully!")


st.subheader("Enter Customer Information")

with st.form("customer_form"):
    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Personal Information")

        gender = st.selectbox("Gender", ["Female", "Male"])

        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox("Has Partner?", ["Yes", "No"])

        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

        st.markdown("### Account Information")

        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)

        contract = st.selectbox(
            "Contract Type", ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with col2:
        st.markdown("### Services Information")

        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

    st.markdown("### Billing Information")

    col3, col4 = st.columns(2)

    with col3:
        monthly_charges = st.number_input(
            "Monthly Charges", min_value=0.0, max_value=150.0, value=70.0, step=1.0
        )

    with col4:
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, max_value=10000.0, value=840.0, step=10.0
        )

    # Submit button
    submitted = st.form_submit_button("Predict Churn", use_container_width=True)


if submitted:
    # Create a DataFrame from customer inputs
    customer_data = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )

    st.write("Customer Information:")
    st.dataframe(customer_data)

    # Make prediction
    prediction = model.predict(customer_data)[0]

    # Get churn probability
    probability = model.predict_proba(customer_data)[0][1]

    # Display result
    st.subheader("Prediction Result")

    # Determine risk category
    if probability < 0.30:
        risk_category = "Low Risk"

    elif probability < 0.60:
        risk_category = "Medium Risk"

    else:
        risk_category = "High Risk"

    # Display prediction
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

    st.metric(label="Churn Probability", value=f"{probability * 100:.2f}%")

    st.metric(label="Churn Risk Category", value=risk_category)

    st.progress(float(probability))

    st.metric(label="Churn Probability", value=f"{probability * 100:.2f}%")

    st.progress(float(probability))

    prediction = model.predict(customer_data)[0]

    probability = model.predict_proba(customer_data)[0][1]

    st.subheader("Probability Breakdown")

    probability_data = pd.DataFrame(
        {
            "Customer Status": ["Stay", "Churn"],
            "Probability": [1 - probability, probability],
        }
    )

    st.bar_chart(probability_data.set_index("Customer Status"))

    st.subheader("Customer Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Contract Type", contract)

    with col2:
        st.metric("Tenure", f"{tenure} Months")

    with col3:
        st.metric("Monthly Charges", f"${monthly_charges:.2f}")

    # Prepare prediction report
    report = customer_data.copy()

    report["Predicted_Churn"] = "Yes" if prediction == 1 else "No"

    report["Churn_Probability"] = round(probability * 100, 2)

    report["Risk_Category"] = risk_category

    # Download button
    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction Report",
        data=csv,
        file_name="customer_churn_prediction.csv",
        mime="text/csv",
    )
