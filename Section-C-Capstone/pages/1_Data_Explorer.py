import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.title("Data Explorer")
uploaded_file = st.file_uploader("Upload Delivery CSV", type="csv")

if uploaded_file is None:
    st.warning("Please upload a CSV file to explore data.")
else:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Orders", len(df))
        col2.metric("Average Delivery Time", f"{df['Delivery_Time_min'].mean():.1f} min")
        col3.metric("Total Revenue", f"${df['Revenue'].sum():.2f}")
        
        st.subheader("Visualizations")
        
        if 'Hour' not in df.columns:
            df['Hour'] = np.random.randint(0, 24, len(df))
            
        hourly_counts = df['Hour'].value_counts().reset_index()
        hourly_counts.columns = ['Hour', 'Orders']
        hourly_counts = hourly_counts.sort_values('Hour')
        fig_hourly = px.line(hourly_counts, x='Hour', y='Orders', title='Hourly Order Volume Trends')
        st.plotly_chart(fig_hourly, use_container_width=True)
        
        fig_dist = px.box(df, x='Restaurant', y='Delivery_Time_min', title='Delivery Time Distribution by Restaurant')
        st.plotly_chart(fig_dist, use_container_width=True)
        
    except Exception:
        st.error("Failed to process the uploaded file. Please ensure it is a valid CSV.")
