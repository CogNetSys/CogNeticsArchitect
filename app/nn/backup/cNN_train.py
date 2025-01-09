import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the generated data
df = pd.read_csv("synthetic_training_data.csv")

# Preprocess the data
X = df[["Year", "Quarter", "Requests"]].values  # Features
y = df["Class"].values  # Labels

# Encode labels as numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert "high", "normal", "low" to 2, 1, 0

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Convert labels to categorical (one-hot encoding)
y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Build the neural network model
model = Sequential(
    [
        Dense(64, activation="relu", input_shape=(3,)),
        Dense(64, activation="relu"),
        Dense(3, activation="softmax"),  # 3 output classes: low, normal, high
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(
    X_train,
    y_train_categorical,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test_categorical),
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save("demand_classification_nn.h5")
print("Model saved as 'demand_classification_nn.h5'")

# Predict on new data
new_data = np.array([[2024, 1, 1800]])  # Example: Q1 2024 with 1800 requests
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
print(f"Predicted Class for new data: {predicted_class[0]}")
