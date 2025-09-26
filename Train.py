import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf
import joblib

df = pd.read_csv(r"C:\Users\Vedant Maladkar\Downloads\Crop_recommendation.csv")

X = df.drop('label', axis=1)
y = df['label']

le_cr = LabelEncoder()
y = le_cr.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model=tf.keras.Sequential([
    tf.keras.layers.Dense(512, input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dense(8),
    tf.keras.layers.Dense(4),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1)

model.save("crop_model.keras")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_cr, "le_crop.pkl")

print("Model saved successfully!")
