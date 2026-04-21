import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
from src.eda import run_eda

#  LOAD MODELS
clf = joblib.load("models/classifier_pipeline.pkl")
reg = joblib.load("models/regressor_pipeline.pkl")
expected_cols = joblib.load("models/feature_columns.pkl")

#  LOAD DATASET (for EDA)
df = pd.read_csv("data/india_housing_prices.csv")

#  SIDEBAR NAVIGATION (eda)
page = st.sidebar.selectbox("Choose Page", ["Prediction", "EDA"])

# =========================================================
#  PREDICTION PAGE
# =========================================================
if page == "Prediction":

    st.title(" Real Estate Investment Advisor")

    # Inputs
    state = st.text_input("State", "Maharashtra")
    city = st.text_input("City", "Mumbai")
    ptype = st.selectbox("Property Type", ["Apartment", "Villa", "House"])

    bhk = st.number_input("BHK", 1, 10, 2)
    sqft = st.number_input("Size", 500, 10000, 1000)
    price = st.number_input("Price", 10, 10000, 100)
    year = st.number_input("Year Built", 1990, 2026, 2015)

    furnished = st.selectbox("Furnished", ["Unfurnished", "Semi", "Fully"])
    floor = st.number_input("Floor", 0, 50, 2)
    total = st.number_input("Total Floors", 1, 100, 10)

    schools = st.slider("Schools", 0, 10, 3)
    hospitals = st.slider("Hospitals", 0, 10, 3)
    transport = st.slider("Transport", 0, 10, 5)

    parking = st.slider("Parking", 0, 5, 1)
    security = st.selectbox("Security", ["Yes", "No"])
    amenities = st.selectbox("Amenities", ["Basic", "Premium"])
    facing = st.selectbox("Facing", ["North", "South", "East", "West"])
    owner = st.selectbox("Owner", ["Individual", "Builder"])
    status = st.selectbox("Status", ["Available", "Sold"])

    # Feature engineering
    age = 2026 - year
    pps = price / sqft if sqft != 0 else 0

    input_df = pd.DataFrame([{
        "State": state,
        "City": city,
        "Property_Type": ptype,
        "BHK": bhk,
        "Size_in_SqFt": sqft,
        "Price_in_Lakhs": price,
        "Price_per_SqFt": pps,
        "Year_Built": year,
        "Age_of_Property": age,
        "Furnished_Status": furnished,
        "Floor_No": floor,
        "Total_Floors": total,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Amenities": amenities,
        "Facing": facing,
        "Owner_Type": owner,
        "Availability_Status": status
    }])

    # Align
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = "Unknown"

    input_df = input_df[expected_cols]

    # Predict
    if st.button("Predict"):
        pred = clf.predict(input_df)[0]
        future = reg.predict(input_df)[0]

        if pred == 1:
            st.success("✅ Good Investment")
        else:
            st.error("❌ Not Good Investment")

        st.write(f"Future Price: ₹{future:.2f} Lakhs")


# =========================================================
#  EDA PAGE
# =========================================================
elif page == "EDA":

    st.title(" Exploratory Data Analysis")

    # Basic info
    st.write("### Dataset Overview")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.dataframe(df.head())

    # Run EDA graphs
    run_eda(df)
