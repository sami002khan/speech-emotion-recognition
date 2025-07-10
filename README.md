# speech-emotion-recognition
Speech Emotion Recognition using MFCC features and a CNN model trained on the RAVDESS dataset.

📌 Objective
The objective of this project is to develop a machine learning model that recognizes human emotions from speech using audio signals. The model leverages Mel Frequency Cepstral Coefficients (MFCCs) for feature extraction and uses a Convolutional Neural Network (CNN) to classify emotions.

📁 Dataset
Name: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Type: Speech-only, WAV audio files

Actors: 24 (12 male, 12 female)

Classes:

Neutral (01)

Calm (02)

Happy (03)

Sad (04)

Angry (05)

Fearful (06)

Disgust (07)

Surprised (08)

⚙️ Technologies Used
Language: Python

Libraries:

TensorFlow/Keras

Librosa

NumPy

Scikit-learn

Matplotlib

🧠 Modeling Steps
✅ Step 1: Load Dataset
RAVDESS audio files were unzipped and read from directory.

✅ Step 2: Preprocessing
Audio files trimmed to 3 seconds with 0.5s offset to avoid silence.

Resampled and normalized using librosa.

✅ Step 3: Feature Extraction
MFCC features extracted with 40 coefficients.

Mean pooling applied to each feature vector → final shape: (40,)

✅ Step 4: Label Encoding
Emotions decoded from filename format (e.g., 03 = happy).

Encoded into integers using LabelEncoder.

✅ Step 5: Data Splitting
Stratified split into:

64% Training

16% Validation

20% Testing

✅ Step 6: Build Model
1D CNN architecture:

Conv1D + ReLU

MaxPooling

Dropout (40%)

Dense + Softmax output

Optimizer: Adam with learning rate = 0.0005

✅ Step 7: Train Model
Trained for 50 epochs with EarlyStopping on validation loss.

Batch size: 32

✅ Step 8: Evaluate Model
Evaluated on test set using:

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

✅ Step 9: Save Model
Saved trained model to: emotion_recognition_model.h5

Saved LabelEncoder to: label_encoder.pkl

📊 Results
Metric	Value
Test Accuracy	~88%
Model Type	1D CNN
Input Shape	(40, 1)
Classes	8 emotions

✅ Conclusion
The model effectively classifies speech-based emotions with high accuracy using MFCC features and a CNN model. The system can be used for human-computer interaction, emotion-aware apps, and voice-based assistants.

