import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Convert text to numbers
encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    encoders[column] = le

# Split features and target
X = data.drop("risk", axis=1)
y = data["risk"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model + encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

print("Model trained and saved successfully!")