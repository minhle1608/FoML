import streamlit as st
import requests

st.title('Predicting car price')
st.write("Enter the car you want to predict the price")

brand_input = st.text_input("Enter a car brand")

year = st.slider("Car Year", 1994, 2020, 2015)
trans_map = {'Manual': 0, 'Automatic': 1}
transmission = st.selectbox("Transmission", list(trans_map.keys()))
transmission_val = trans_map[transmission]

seller_map = {'Individual': 0, 'Trustmark Dealer': 1, 'Dealer': 2}
seller_type = st.selectbox("Seller", list(seller_map.keys()))
seller_val = seller_map[seller_type]

fuel_map = {'CNG': 0, 'LPG': 1, 'Petrol': 2, 'Diesel': 3}
fuel_type = st.selectbox("Fuel", list(fuel_map.keys()))
fuel_val = fuel_map[fuel_type]

owner_map = {'Fourth and above': 0, 'Third': 1, 'Second': 2, 'First': 3, 'Test Drive': 4}
owner_type = st.selectbox("Owner", list(owner_map.keys()))
owner_val = owner_map[owner_type]

engine = st.number_input("Input engine (CC) from 624 - 3604")
horsepower = st.number_input("Input engine horsepower (bph) from 33 - 400")

API_URL = "http://localhost:8000/predict"

if st.button("Predict"):
    if not brand_input:
        st.warning("Please input a brand")
    else:
        user_input = {
            "brand": "name_" + brand_input,
            "seller_type": seller_val,
            "fuel_type": fuel_val,
            "owner": owner_val,
            "year": year,
            "transmission": transmission_val,
            "engine": engine,
            "max_power": horsepower
        }
    
    try:
        response = requests.post(API_URL, json=user_input)

        if response.status_code == 200:
                result = response.json()
                price = result['prediction']
                st.success(f"Estimated Price (converted to USD): ${(price*0.011):,.2f}")
        else:
            st.error(f"Error from API: {response.text}")

    except requests.exceptions.ConnectionError:
            st.error("Cannot connect")