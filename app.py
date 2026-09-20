import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Second-Hand Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)


# --------------------------------------------------
# Load Model and Preprocessing Objects
# --------------------------------------------------

model = tf.keras.models.load_model(
    "model/ann_model.h5",
    compile=False
)

scaler = joblib.load("model/scaler.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")


# --------------------------------------------------
# Application Title
# --------------------------------------------------

st.title("🚗 Second-Hand Car Price Prediction")

st.write(
    "Enter the details of a second-hand car to predict "
    "its estimated selling price."
)


# --------------------------------------------------
# User Input
# --------------------------------------------------

manufacturer = st.selectbox(
    "Manufacturer",
    [
        "Mercedes-Benz",
        "Toyota",
        "Audi",
        "Nissan",
        "Volkswagen",
        "Chevrolet",
        "BMW",
        "Tesla",
        "Honda",
        "Ford"
    ]
)


model_name = st.selectbox(
    "Model",
    [
        "Cruze",
        "A4",
        "C-Class",
        "Model 3",
        "Golf",
        "Fiesta",
        "Civic",
        "Altima",
        "3 Series",
        "Corolla"
    ]
)


engine_size = st.number_input(
    "Engine Size (L)",
    min_value=0.1,
    max_value=5.0,
    value=2.0,
    step=0.1
)


fuel_type = st.selectbox(
    "Fuel Type",
    [
        "Electric",
        "Petrol",
        "Diesel",
        "Hybrid"
    ]
)


year = st.number_input(
    "Year of Manufacture",
    min_value=1950,
    max_value=2026,
    value=2018,
    step=1
)


mileage = st.number_input(
    "Mileage",
    min_value=0,
    max_value=300000,
    value=50000,
    step=1000
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Price"):

    # Create DataFrame from user input
    input_data = pd.DataFrame({
        "Manufacturer": [manufacturer],
        "Model": [model_name],
        "Engine Size (L)": [engine_size],
        "Fuel Type": [fuel_type],
        "Year of Manufacture": [year],
        "Mileage": [mileage]
    })


    # --------------------------------------------------
    # One-Hot Encoding
    # --------------------------------------------------

    input_encoded = pd.get_dummies(
        input_data,
        columns=[
            "Manufacturer",
            "Model",
            "Fuel Type"
        ],
        drop_first=False,
        dtype=int
    )


    # --------------------------------------------------
    # Match Training Features
    # --------------------------------------------------

    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # --------------------------------------------------
    # Scale Numerical Features
    # --------------------------------------------------

    numeric_columns = [
        "Engine Size (L)",
        "Year of Manufacture",
        "Mileage"
    ]

    input_encoded[numeric_columns] = scaler.transform(
        input_encoded[numeric_columns]
    )


    # --------------------------------------------------
    # Make Prediction
    # --------------------------------------------------

    prediction = model.predict(input_encoded)

    predicted_price = float(prediction[0][0])


    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.success(
        f"Estimated Car Price: £{predicted_price:,.2f}"
    )
