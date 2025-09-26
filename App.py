import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import requests
import random

model = tf.keras.models.load_model("crop_model.keras")
scaler = joblib.load("scaler.pkl")
le_cr = joblib.load("le_crop.pkl")

API_KEY = "1fbd77840356c18b48a0fac06596fa09"

DEFAULT_SOIL_LIST = [
    {
        "n": round(random.uniform(0.1, 0.5), 2),  # Nitrogen in %
        "p": round(random.uniform(5, 20), 2),     # Phosphorus in mg/kg
        "k": round(random.uniform(100, 200), 2),  # Potassium in mg/kg
        "ph": round(random.uniform(5.5, 7.5), 2)}]

def get_lat_lon(city):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
    resp = requests.get(url)
    if resp.status_code == 200 and len(resp.json()) > 0:
        data = resp.json()[0]
        return float(data["lat"]), float(data["lon"])
    return None, None

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        rainfall = data.get('rain', {}).get('1h', 0)
        return temp, humidity, rainfall
    return None, None, None

def fetch_soil_data(lat: float, lon: float):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&depth=0-30cm&value=mean"
    resp = requests.get(url)
    if resp.status_code != 200:
        return {}
    data = resp.json()
    layers = data.get("properties", {}).get("layers", [])
    soil = {}
    for layer in layers:
        name = layer.get("name", "").lower()
        for d in layer.get("depths", []):
            val = d.get("values", {}).get("mean", None)
            if val is not None:
                if "phh2o" in name and "ph" not in soil:
                    soil["ph"] = val / 10
                elif "nitrogen" in name and "n" not in soil:
                    soil["n"] = val
                elif "phosphorus" in name and "p" not in soil:
                    soil["p"] = val
                elif "potassium" in name and "k" not in soil:
                    soil["k"] = val
    return soil

st.set_page_config(page_title=" AI Crop Recommendation", layout="centered")
st.title(" AI-Based Crop Recommendation System")
st.subheader("Farmer-Friendly using Real-Time Weather + Soil Intelligence")

city = st.text_input(" Enter your City / District Name", placeholder="e.g., Ludhiana")

if st.button(" Recommend Best Crop"):
    if city.strip() == "":
        st.error(" Please enter a valid city name.")
    else:
        # 1️⃣ Get lat/lon
        lat, lon = get_lat_lon(city)
        if lat is None or lon is None:
            st.warning(" Could not find city coordinates. Using random default soil values.")
            soil = random.choice(DEFAULT_SOIL_LIST)
        else:
            st.write(f"DEBUG: lat={lat}, lon={lon}")
            soil = fetch_soil_data(lat, lon)
            if not all(k in soil for k in ["n", "p", "k", "ph"]):
                default_soil = random.choice(DEFAULT_SOIL_LIST)
                for key in ["n", "p", "k", "ph"]:
                    if key not in soil:
                        soil[key] = default_soil[key]

        st.success(f" Soil data: N={soil['n']:.2f} | P={soil['p']:.2f} | K={soil['k']:.2f} | pH={soil['ph']:.2f}")

        temperature, humidity, rainfall = get_weather(city)
        if temperature is None:
            st.warning(" Weather data not available. Using defaults (25°C, 60% humidity, 0mm rainfall).")
            temperature, humidity, rainfall = 25, 60, 0
        else:
            st.success(f" Weather: Temperature: {temperature}°C | Humidity: {humidity}% |  Rainfall: {rainfall} mm")

        input_data = np.array([[soil['n'], soil['p'], soil['k'], temperature, humidity, soil['ph'], rainfall]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        crop_index = np.argmax(prediction)
        crop_name = le_cr.inverse_transform([crop_index])[0]
        confidence = np.max(prediction) * 100

        st.markdown(" Recommended Crop:")
        st.success(f" Recommended: {crop_name}")
        st.info(f"Confidence: {confidence:.2f}%")
