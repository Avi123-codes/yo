#  Imports
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import joblib

#  Loading CSV from Google Drive
df = pd.read_csv("/content/drive/MyDrive/asl_alphabet_full.csv")

#  Split into features and labels
X = df.drop("label", axis=1).values.astype("float32")
y = df["label"].values

# Encode labels a becomes 0, space becomes 26
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)


joblib.dump(encoder, "asl_label_encoder.pkl")

# MAking it greyscale
X = X.reshape(-1, 64, 64, 1)

#  Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=3)

# CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 1)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(y_onehot.shape[1], activation='softmax')  # output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#  Training model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

#  Saving model
model.save("/content/drive/MyDrive/asl_cnn_model.keras")  
print("✅ Model saved to your Drive!")

# Evaluating model
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"\n🎯 Test Accuracy: {accuracy*100:.2f}%")
