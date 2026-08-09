import streamlit as st
import tensorflow as tf
import plotly.express as px
import pandas as pd
import numpy as np
import os

@st.cache_resource
def load_keras_model():
    if os.path.exists('demand_model.keras'):
        return tf.keras.models.load_model('demand_model.keras')
    return None

st.title("Model Architecture & History")
model = load_keras_model()

if model is None:
    st.error("Model file 'demand_model.keras' not found.")
else:
    st.subheader("Architecture")
    layer_data = []
    for layer in model.layers:
        try:
            out_shape = str(layer.output_shape)
        except AttributeError:
            try:
                out_shape = str(layer.output.shape)
            except Exception:
                out_shape = "Unknown"
                
        layer_data.append({
            "Layer Name": layer.name,
            "Output Shape": out_shape,
            "Parameters": layer.count_params()
        })
    st.table(pd.DataFrame(layer_data))
    
    st.subheader("Training History")
    st.info("Displaying synthetic training history as Keras doesn't save history in the .keras file.")
    epochs = np.arange(1, 31)
    train_acc = np.linspace(0.4, 0.95, 30) + np.random.normal(0, 0.02, 30)
    val_acc = np.linspace(0.4, 0.85, 30) + np.random.normal(0, 0.03, 30)
    train_loss = np.linspace(2.0, 0.2, 30) + np.random.normal(0, 0.05, 30)
    val_loss = np.linspace(2.0, 0.5, 30) + np.random.normal(0, 0.05, 30)
    
    hist_df = pd.DataFrame({
        "Epoch": epochs,
        "Train Accuracy": train_acc,
        "Validation Accuracy": val_acc
    })
    fig_acc = px.line(hist_df, x="Epoch", y=["Train Accuracy", "Validation Accuracy"], title="Model Accuracy")
    st.plotly_chart(fig_acc)
    
    loss_df = pd.DataFrame({
        "Epoch": epochs,
        "Train Loss": train_loss,
        "Validation Loss": val_loss
    })
    fig_loss = px.line(loss_df, x="Epoch", y=["Train Loss", "Validation Loss"], title="Model Loss")
    st.plotly_chart(fig_loss)
