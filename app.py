import streamlit as st
import pickle
import pandas as pd

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

st.title("Context-Aware Pedestrian Risk Analyzer")

walking_speed = st.selectbox("Walking speed", ["slow", "medium", "fast"])
phone_usage = st.selectbox("Using phone?", ["yes", "no"])
head_angle = st.selectbox("Head angle", ["low", "medium", "high"])
traffic_density = st.selectbox("Traffic density", ["low", "medium", "high"])

if st.button("Analyze Risk"):

    data = pd.DataFrame([[walking_speed, phone_usage, head_angle, traffic_density]],
                        columns=["walking_speed","phone_usage","head_angle","traffic_density"])

    for column in data.columns:
        data[column] = encoders[column].transform(data[column])

    prediction = model.predict(data)[0]
    risk_label = encoders["risk"].inverse_transform([prediction])[0]

    st.subheader(f"Predicted Risk Level: {risk_label.upper()}")

    if risk_label == "dangerous":
        st.error("⚠️ Suggested Action: Trigger vibration alert")
    else:
        st.success("✅ Safe walking condition")