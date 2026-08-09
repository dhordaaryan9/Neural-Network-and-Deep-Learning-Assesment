import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.express as px
import pandas as pd

@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('demand_model.keras')

st.title("Order Demand Prediction")

hour = st.slider("Hour of Day", 0, 23, 12)
day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
temp = st.number_input("Temperature (°C)", min_value=10.0, max_value=45.0, value=25.0)

day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
classes = ["Low", "Medium", "High"]

if st.button("Predict Demand"):
    try:
        model = load_keras_model()
        encoded_day = day_mapping[day]
        features = np.array([[hour, encoded_day, temp]])
        
        probs = model.predict(features)[0]
        pred_idx = np.argmax(probs)
        pred_class = classes[pred_idx]
        confidence = probs[pred_idx] * 100
        
        if pred_class == "High":
            st.warning(f"Predicted Demand: {pred_class} ({confidence:.1f}% confidence)")
        else:
            st.success(f"Predicted Demand: {pred_class} ({confidence:.1f}% confidence)")
            
        plot_df = pd.DataFrame({"Demand Class": classes, "Probability": probs})
        fig = px.bar(plot_df, x="Demand Class", y="Probability", title="Model Confidence Scores")
        st.plotly_chart(fig)
        
    except Exception:
        st.error("Failed to load 'demand_model.keras' or process prediction.")
