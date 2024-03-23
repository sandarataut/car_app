import pandas as pd
import streamlit as st 
import joblib
import numpy as np

with open('carmodel.joblib', 'rb') as f:
    model = joblib.load(f)

list_CarNames = {90.0: "ritz", 93.0: "sx4", 96.0: "wagon r", 92.0: "swift", 1.0: "Honda CB Trigger"}
list_FuelTypes = {2.0: "Petrol", 1.0: "Diesel", 0.0: "CNG"}
list_SellerTypes = {0.0: "Dealer", 1.0: "Individual"}
list_Transmissions = {1.0: "Manual", 0.0: "Automatic"}
list_Owner = {1: "Yes", 0: "No"}

def format_CarNames(option):
    return list_CarNames[option]

def format_FuelTypes(option):
    return list_FuelTypes[option]

def format_SellerTypes(option):
    return list_SellerTypes[option]

def format_Transmissions(option):
    return list_Transmissions[option]

def format_Owner(option):
    return list_Owner[option]

def main():
    car_Name = st.sidebar.selectbox("Car Name", options=list(list_CarNames.keys()), format_func=format_CarNames)
    year = st.sidebar.slider('Year', 2000, 2024, 2010)
    present_Price = st.sidebar.slider('Present Price', 0.3, 100.0, 10.0)
    kms_Driven = st.sidebar.slider('Kilometer Driven', 1000.00, 100000.00, 5000.00)
    fuel_Type = st.sidebar.selectbox("Fuel Type", options=list(list_FuelTypes.keys()), format_func=format_FuelTypes)
    seller_Type = st.sidebar.selectbox("Seller Type", options=list(list_SellerTypes.keys()), format_func=format_SellerTypes)
    transmission = st.sidebar.selectbox("Transmission", options=list(list_Transmissions.keys()), format_func=format_Transmissions)
    owner = st.sidebar.selectbox("Owner", options=list(list_Owner.keys()), format_func=format_Owner)

    # Getting Prediction from model
    inp = np.array([[car_Name,year,present_Price,kms_Driven,fuel_Type,seller_Type,transmission, owner]])
    prediction = model.predict(inp)

    ## Show Results when prediction is done
    if prediction.any():
        st.write('''
        ## Results
        Following is the prediction result for selling price:
        ''')
        
        result = prediction
        st.write("Predicted selling price " + str(result))

if __name__ == "__main__":
    main()
