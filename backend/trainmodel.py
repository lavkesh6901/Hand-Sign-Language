import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv(r"D:\Projects\protottype\backend\landmarks_dataset.csv")

# Split features and labels
X = df.drop("label", axis=1).values
y = df["label"].values

# Encode labels to numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save class labels
with open(r"D:\Projects\protottype\backend\models\labels.txt", "w") as f:
    for i in encoder.classes_:
        f.write(i + "\n")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(42,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Save model
model.save(r"D:\Projects\protottype\backend\models/landmark_model.keras")
print("✅ Model trained and saved to models/landmark_model.keras")
