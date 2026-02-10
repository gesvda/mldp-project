import joblib
import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
model = joblib.load("hdb_rf_model.pkl")

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("🏠 Singapore HDB Resale Price Predictor")

st.write(
    "Enter flat details below to estimate resale price using a tuned Random Forest model."
)

# -----------------------------
# INPUT OPTIONS
# -----------------------------
towns = [
    "ANG MO KIO","BEDOK","BISHAN","CHOA CHU KANG",
    "HOUGANG","JURONG WEST","PUNGGOL","SENGKANG",
    "TAMPINES","WOODLANDS","YISHUN"
]

flat_types = [
    "2 ROOM","3 ROOM","4 ROOM","5 ROOM",
    "EXECUTIVE","MULTI-GENERATION"
]

# -----------------------------
# USER INPUTS
# -----------------------------
town_selected = st.selectbox("Select Town", towns)

flat_type_selected = st.selectbox("Select Flat Type", flat_types)

floor_area_selected = st.slider(
    "Select Floor Area (sqm)",
    min_value=30,
    max_value=200,
    value=90
)

remaining_lease_selected = st.slider(
    "Remaining Lease (years)",
    min_value=0,
    max_value=99,
    value=75
)

lease_commence_selected = st.number_input(
    "Lease Commence Year",
    min_value=1960,
    max_value=2023,
    value=2000
)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict HDB Price"):

    # Create dataframe from inputs
    df_input = pd.DataFrame({
        "floor_area_sqm": [floor_area_selected],
        "remaining_lease": [remaining_lease_selected],
        "lease_commence_date": [lease_commence_selected],
        "flat_type": [flat_type_selected],
        "town": [town_selected]
    })

    # One-hot encoding
    df_input = pd.get_dummies(
        df_input,
        columns=["flat_type", "town"]
    )

    # Align with model training columns
    df_input = df_input.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Predict
    prediction = model.predict(df_input)[0]

    st.success(f"Estimated Resale Price: ${prediction:,.0f}")

# -----------------------------
# PAGE DESIGN (BACKGROUND)
# -----------------------------
st.markdown("""
<style>

/* App background */
.stApp {
    background-image: url("https://images.openai.com/static-rsc-3/tDkkSA7wXJ9Dd0TjPlFu1b9dxdNuXd-L0W9co-_J5BUfpk-s2WaBa0mwDzTi6d2GQ-aSsXs6iWWyKj2xWHHJq9SRI7EwsrvJ-RoLHCTFVpU?purpose=fullsize&v=1");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Dark glass container */
section.main > div {
    background-color: rgba(0, 0, 0, 0.70);
    padding: 2.5rem;
    border-radius: 20px;
}

/* Make ALL text white */
h1, h2, h3, h4, h5, h6, p, label, div {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)



