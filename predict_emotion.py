import librosa
import numpy as np
import tensorflow as tf
import pickle

# Load model and encoder
model = tf.keras.models.load_model("emotion_recognition_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Predict function
def predict_emotion(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0).reshape(1, 40, 1)
    pred = model.predict(mfcc_scaled)
    emotion = le.inverse_transform([np.argmax(pred)])
    return emotion[0]

# Example
# print(predict_emotion("sample.wav"))
