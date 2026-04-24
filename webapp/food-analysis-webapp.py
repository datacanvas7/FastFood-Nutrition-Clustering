import streamlit as st
import pandas as pd
import joblib
import json

# Load models
kmeans = joblib.load('models/kmeans_clustering_model.pkl')
scaler = joblib.load('models/standard_scaler.pkl')

with open('models/feature_names.json', 'r') as f:
    features = json.load(f)

# Cluster labels (business meaning)
cluster_map = {
    0: "High-Protein, Low-Carb",
    1: "High-Carb",
    2: "Balanced",
    3: "High-Nutrient Density",
    4: "High-Calorie, High-Fat"
}

# App title
st.title("🍔 Fast Food Nutrition Clustering")
st.write("Predict which nutritional cluster a food item belongs to")

st.sidebar.header("Input Nutritional Values")

# Create inputs dynamically
user_input = {}
for feature in features:
    user_input[feature] = st.sidebar.number_input(feature, value=0.0)

# Predict button
if st.sidebar.button("Predict Cluster"):
    
    # Convert to DataFrame
    df = pd.DataFrame([user_input])
    
    # Scale
    scaled = scaler.transform(df)
    
    # Predict
    cluster = int(kmeans.predict(scaled)[0])
    
    # Output
    st.subheader("📊 Prediction Result")
    st.success(f"Cluster: {cluster}")
    st.info(f"Category: {cluster_map.get(cluster, 'Unknown')}")

    # Show input summary
    st.subheader("🧾 Input Summary")
    st.dataframe(df)