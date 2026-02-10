import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Singapore HDB Predictor",
    page_icon="🏠",
    layout="centered"
)


# -------------------------------------------------
# LOAD MODEL (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_model():

    MODEL_URL = "https://drive.google.com/uc?id=1X8kywjv2nvbrsxf_4ZueCw9vNzeLkhU6"
    MODEL_PATH = "hdb_rf_model.pkl"

    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    return joblib.load(MODEL_PATH)


model = load_model()


# -------------------------------------------------
# BUILD OPTIONS DIRECTLY FROM MODEL
# (Prevents feature mismatch FOREVER)
# -------------------------------------------------
model_columns = model.feature_names_in_

towns = sorted([
    col.replace("town_", "")
    for col in model_columns
    if col.startswith("town_")
])

flat_types = sorted([
    col.replace("public_housing_flat_type_", "")
    for col in model_columns
    if col.startswith("public_housing_flat_type_")
])


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🏠 Singapore HDB Resale Price Predictor")

st.write(
    "Enter flat details below to estimate resale price using a "
    "Random Forest machine learning model."
)


town_selected = st.selectbox("Select Town", towns)

flat_type_selected = st.selectbox("Select Flat Type", flat_types)

floor_area_selected = st.slider(
    "Floor Area (sqm)",
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


# -------------------------------------------------
# PREDICTION (PRODUCTION SAFE)
# -------------------------------------------------
if st.button("Predict Resale Price"):

    # Create EMPTY dataframe matching model EXACTLY
    df_input = pd.DataFrame(
        np.zeros((1, len(model_columns))),
        columns=model_columns
    )

    # Fill numeric values
    df_input.loc[0, "floor_area_sqm"] = floor_area_selected
    df_input.loc[0, "remaining_lease"] = remaining_lease_selected
    df_input.loc[0, "lease_commence_date"] = lease_commence_selected

    # Activate one-hot columns safely
    town_col = f"town_{town_selected}"
    flat_col = f"public_housing_flat_type_{flat_type_selected}"

    if town_col in df_input.columns:
        df_input.loc[0, town_col] = 1

    if flat_col in df_input.columns:
        df_input.loc[0, flat_col] = 1

    # Predict
    prediction = model.predict(df_input)[0]

    st.metric(
        label="Estimated HDB Resale Price",
        value=f"${prediction:,.0f}"
    )


# -------------------------------------------------
# BACKGROUND (Nice but optional)
# -------------------------------------------------
st.markdown("""
<style>

.stApp {
    background-image: url("https://images.unsplash.com/photo-1598928506311-c55ded91a20c?q=80&w=2070");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

section.main > div {
    background-color: rgba(0, 0, 0, 0.72);
    padding: 2.5rem;
    border-radius: 18px;
}

h1, h2, h3, h4, h5, h6, p, label, div {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)
