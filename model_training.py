import os
import numpy as np
import librosa
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Emotion map (RAVDESS)
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# MFCC extractor
def extract_features(file_path, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except:
        return None

# Load data
X, y = [], []
data_path = 'ravdess_data/audio_speech_actors_01-24'

for subdir, _, files in os.walk(data_path):
    for file in files:
        if file.endswith('.wav'):
            code = file.split('-')[2]
            label = emotion_map.get(code)
            if label:
                path = os.path.join(subdir, file)
                feat = extract_features(path)
                if feat is not None:
                    X.append(feat)
                    y.append(label)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset
X = np.array(X)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

# Reshape for CNN
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# CNN Model
model = Sequential([
    Conv1D(128, 3, activation='relu', input_shape=(40, 1)),
    MaxPooling1D(2),
    Dropout(0.4),
    Conv1D(256, 3, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# Train
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stop])

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")

# Save model
model.save("emotion_recognition_model.h5")
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
