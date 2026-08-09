import streamlit as st
import pandas as pd

st.title("Delivery Summary Dashboard")

uploaded_file = st.file_uploader("Upload Delivery CSV", type="csv")

if uploaded_file is None:
    st.info("Please upload delivery data CSV file to view the dashboard.")
else:
    df = pd.read_csv(uploaded_file)
    
    cities = df['City'].unique()
    selected_city = st.sidebar.selectbox("Select City", cities)
    
    filtered_df = df[df['City'] == selected_city]
    
    total_orders = len(filtered_df)
    avg_delivery_time = filtered_df['Delivery_Time_min'].mean()
    total_revenue = filtered_df['Revenue'].sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Orders", total_orders)
    col2.metric("Average Delivery Time (min)", f"{avg_delivery_time:.1f}")
    col3.metric("Total Revenue", f"${total_revenue:.2f}")
    
    restaurant_counts = filtered_df['Restaurant'].value_counts()
    st.bar_chart(restaurant_counts)
