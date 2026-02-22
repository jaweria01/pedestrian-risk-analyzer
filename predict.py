import pickle
import pandas as pd

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

print("Enter pedestrian context:")

walking_speed = input("Walking speed (slow/medium/fast): ")
phone_usage = input("Using phone? (yes/no): ")
head_angle = input("Head angle (low/medium/high): ")
traffic_density = input("Traffic density (low/medium/high): ")

# Create dataframe
data = pd.DataFrame([[
    walking_speed,
    phone_usage,
    head_angle,
    traffic_density
]], columns=[
    "walking_speed",
    "phone_usage",
    "head_angle",
    "traffic_density"
])

# Encode input
for column in data.columns:
    data[column] = encoders[column].transform(data[column])

# Predict
prediction = model.predict(data)[0]
risk_label = encoders["risk"].inverse_transform([prediction])[0]

print("\nPredicted Risk Level:", risk_label.upper())

if risk_label == "dangerous":
    print("⚠️ Suggested Action: Trigger vibration alert")
else:
    print("✅ Safe walking condition")