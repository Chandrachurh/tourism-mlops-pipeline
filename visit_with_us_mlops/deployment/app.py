import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="chandrachurhghosh/tourism_project_rf", filename="best_TourismPackagePrediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Prediction")
st.write("""
Predict whether a customer is likely to purchase a Tourism Package.
Please enter the customer details below to get a prediction.
""")
st.subheader("Enter Customer Details")
# User input
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
CityTier = st.selectbox("CityTier", [1, 2, 3])
MonthlyIncome = st.number_input("MonthlyIncome", min_value=0, value=45000)
NumberOfTrips = st.number_input("NumberOfTrips", min_value=0, value=2)
PitchSatisfactionScore = st.slider("PitchSatisfactionScore", 1, 5, 4)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, value=2)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=1, value=20)
Passport = st.selectbox("Passport", [0, 1])
OwnCar = st.selectbox("OwnCar", [0, 1])
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, value=2)
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, value=0)
PreferredPropertyStar = st.selectbox("PreferredPropertyStar", [3, 4, 5])

# --- Categorical inputs (raw strings; we will one-hot encode) ---
TypeofContact = st.selectbox("TypeofContact", ["Company Invited", "Self Inquiry"])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Self Employed", "Government"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
ProductPitched = st.selectbox("ProductPitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
Designation = st.text_input("Designation", "Executive")

# Predict button
if st.button("Predict Package Purchase"):
    raw_input = {
        "Age": Age,
        "CityTier": CityTier,
        "MonthlyIncome": MonthlyIncome,
        "NumberOfTrips": NumberOfTrips,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "NumberOfFollowups": NumberOfFollowups,
        "DurationOfPitch": DurationOfPitch,
        "Passport": Passport,
        "OwnCar": OwnCar,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "PreferredPropertyStar": PreferredPropertyStar,
        "TypeofContact": TypeofContact,
        "Occupation": Occupation,
        "Gender": Gender,
        "MaritalStatus": MaritalStatus,
        "ProductPitched": ProductPitched,
        "Designation": Designation
    }

    # Convert raw input to DataFrame
    input_df_raw = pd.DataFrame([raw_input])
    st.write("Input DataFrame (Raw):")
    st.dataframe(input_df_raw)

    # The loaded model is a pipeline that handles preprocessing internally
    prediction_score = model.predict(input_df_raw)[0]

    # Apply a threshold to convert the regressor output to a binary prediction
    purchase_prediction = 1 if prediction_score >= 0.5 else 0

    st.subheader("Prediction Result:")
    if purchase_prediction == 1:
        st.success("**Likely to purchase the Wellness Tourism Package!**")
    else:
        st.info("**Not likely to purchase the Wellness Tourism Package.**")
    st.write(f"(Prediction Score: {prediction_score:.4f})")
