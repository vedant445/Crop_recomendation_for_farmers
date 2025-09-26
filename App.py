import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import requests
import random

# Load ML model and encoders
model = tf.keras.models.load_model("crop_model.keras")
scaler = joblib.load("scaler.pkl")
le_cr = joblib.load("le_crop.pkl")

API_KEY = "1fbd77840356c18b48a0fac06596fa09"

DEFAULT_SOIL_LIST = [
    {"n": round(random.uniform(0.1, 0.5), 2),
     "p": round(random.uniform(5, 20), 2),
     "k": round(random.uniform(100, 200), 2),
     "ph": round(random.uniform(5.5, 7.5), 2)}]

# 🌐 Translations
translations = {
    "title": {"en": "AI-Based Crop Recommendation System",
              "hi": "एआई आधारित फसल सिफारिश प्रणाली"},
    "subtitle": {"en": "Farmer-Friendly using Real-Time Weather + Soil Intelligence",
                 "hi": "किसानों के लिए वास्तविक मौसम और मिट्टी पर आधारित"},
    "city_input": {"en": "Enter your City / District Name",
                   "hi": "अपना शहर / जिला दर्ज करें"},
    "placeholder": {"en": "e.g., Ludhiana",
                    "hi": "उदा., लुधियाना"},
    "button": {"en": "Recommend Best Crop",
               "hi": "सर्वश्रेष्ठ फसल सुझाएँ"},
    "soil_data": {"en": "Soil data",
                  "hi": "मिट्टी का डेटा"},
    "weather": {"en": "Weather",
                "hi": "मौसम"},
    "recommended_crop": {"en": "Recommended Crop",
                         "hi": "अनुशंसित फसल"},
    "confidence": {"en": "Confidence",
                   "hi": "विश्वास स्तर"},
    "error_city": {"en": "Please enter a valid city name.",
                   "hi": "कृपया मान्य शहर का नाम दर्ज करें।"},
    "warn_city": {"en": "Could not find city coordinates. Using random default soil values.",
                  "hi": "शहर के निर्देशांक नहीं मिले। डिफ़ॉल्ट मिट्टी के मान उपयोग किए गए।"},
    "warn_weather": {"en": "Weather data not available. Using defaults (25°C, 60% humidity, 0mm rainfall).",
                     "hi": "मौसम का डेटा उपलब्ध नहीं। डिफ़ॉल्ट मान उपयोग किए गए (25°C, 60% नमी, 0mm वर्षा)।"}
}

# 🌐 Language selection
lang = st.sidebar.selectbox("🌍 Language / भाषा", ("English", "हिंदी"))
lang_code = "en" if lang == "English" else "hi"

# 🌐 Utility function for translation
def t(key):
    return translations.get(key, {}).get(lang_code, key)

# 🔹 Page Setup
st.set_page_config(page_title=t("title"), layout="centered")
st.title(t("title"))
st.subheader(t("subtitle"))

city = st.text_input(t("city_input"), placeholder=t("placeholder"))

if st.button(t("button")):
    if city.strip() == "":
        st.error(t("error_city"))
    else:
        # Get coordinates
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
                return data['main']['temp'], data['main']['humidity'], data.get('rain', {}).get('1h', 0)
            return None, None, None

        def fetch_soil_data(lat, lon):
            url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&depth=0-30cm&value=mean"
            resp = requests.get(url)
            if resp.status_code != 200:
                return {}
            data = resp.json()
            soil = {}
            for layer in data.get("properties", {}).get("layers", []):
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

        lat, lon = get_lat_lon(city)
        if lat is None or lon is None:
            st.warning(t("warn_city"))
            soil = random.choice(DEFAULT_SOIL_LIST)
        else:
            soil = fetch_soil_data(lat, lon)
            if not all(k in soil for k in ["n", "p", "k", "ph"]):
                default_soil = random.choice(DEFAULT_SOIL_LIST)
                for key in ["n", "p", "k", "ph"]:
                    if key not in soil:
                        soil[key] = default_soil[key]

        st.success(f"{t('soil_data')}: N={soil['n']:.2f} | P={soil['p']:.2f} | K={soil['k']:.2f} | pH={soil['ph']:.2f}")

        temperature, humidity, rainfall = get_weather(city)
        if temperature is None:
            st.warning(t("warn_weather"))
            temperature, humidity, rainfall = 25, 60, 0
        
        else:
            st.success(f"{t('weather')}: {temperature}°C | {humidity}% | {rainfall} mm")

        input_data = np.array([[soil['n'], soil['p'], soil['k'], temperature, humidity, soil['ph'], rainfall]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        crop_index = np.argmax(prediction)
        crop_name = le_cr.inverse_transform([crop_index])[0]
        confidence = np.max(prediction) * 100

        st.markdown(t("recommended_crop") + ":")
        st.success(f"{t('recommended_crop')}: {crop_name}")
        st.info(f"{t('confidence')}: {confidence:.2f}%")