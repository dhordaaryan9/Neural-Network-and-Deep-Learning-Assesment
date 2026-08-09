import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load('delivery_model.pkl')

st.title("Delivery Time Prediction")

distance = st.slider("Distance (km)", 1, 50, 5)
order_value = st.number_input("Order Value (Rs)", min_value=0.0, value=200.0)
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

time_mapping = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}

if st.button("Predict Delivery Time"):
    try:
        model = load_model()
        encoded_time = time_mapping[time_of_day]
        features = np.array([[distance, order_value, encoded_time]])
        prediction = model.predict(features)[0]
        st.success(f"Predicted delivery time: {prediction:.1f} minutes")
    except Exception:
        st.error("Failed to load the model or process the input. Make sure 'delivery_model.pkl' exists.")
