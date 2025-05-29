import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Title
st.title("🌾 Smart Crop Recommendation System")
st.markdown("Enter soil and environmental conditions to get the best crop recommendation 🌱")

# Load model and dataset
model = load_model("crop_recommendation_model.h5")
df = pd.read_csv("Crop_recommendation.csv")

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Train scaler again
scaler = StandardScaler()
X = df.drop('label', axis=1)
scaler.fit(X)

# Sidebar Inputs
st.sidebar.header("Input Parameters")

def user_input():
    N = st.sidebar.slider("Nitrogen", 0, 140, 90)
    P = st.sidebar.slider("Phosphorus", 5, 145, 42)
    K = st.sidebar.slider("Potassium", 5, 205, 43)
    temperature = st.sidebar.slider("Temperature (°C)", 8.0, 43.0, 20.5)
    humidity = st.sidebar.slider("Humidity (%)", 14.0, 100.0, 82.0)
    ph = st.sidebar.slider("pH", 3.5, 9.5, 6.5)
    rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 200.0)
    data = {
        'N': N, 'P': P, 'K': K,
        'temperature': temperature, 'humidity': humidity,
        'ph': ph, 'rainfall': rainfall
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# Show input
st.subheader("Your Input:")
st.write(input_df)

# Scale and predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
predicted_crop = le.inverse_transform([np.argmax(prediction)])

# Output
st.subheader("🌿 Recommended Crop:")
st.success(predicted_crop[0])

# Optional: Accuracy / Loss Chart
show_graphs = st.checkbox("📈 Show Training Graphs")

if show_graphs:
    st.markdown("### Sample Accuracy/Loss Curves (Simulated for UI)")
    # Fake data just for demo
    acc = np.random.uniform(0.7, 1.0, 50)
    val_acc = acc - np.random.uniform(0.01, 0.05, 50)
    loss = np.random.uniform(0.05, 0.6, 50)
    val_loss = loss + np.random.uniform(0.01, 0.05, 50)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].plot(acc, label="Training Accuracy")
    ax[0].plot(val_acc, label="Validation Accuracy")
    ax[0].legend(); ax[0].set_title("Accuracy")
    ax[1].plot(loss, label="Training Loss")
    ax[1].plot(val_loss, label="Validation Loss")
    ax[1].legend(); ax[1].set_title("Loss")
    st.pyplot(fig)
