# genre_classifier_final.py
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings('ignore')

def is_valid_wav(file_path):
    """Check if .wav file is readable and has audio"""
    try:
        audio, sr = librosa.load(file_path, sr=None, duration=1.0)  # Load 1 sec
        return len(audio) > 0 and not np.isnan(audio).any()
    except:
        return False

print("🔍 Loading and validating dataset...")
X, y = [], []
genres = sorted([g for g in os.listdir("genres_original") if os.path.isdir(f"genres_original/{g}")])

for genre in genres:
    valid_count = 0
    for file in os.listdir(f"genres_original/{genre}"):
        if file.endswith(".wav"):
            file_path = f"genres_original/{genre}/{file}"
            if is_valid_wav(file_path):
                try:
                    # Load full 30 seconds
                    audio, sr = librosa.load(file_path, sr=22050, duration=30.0)
                    # Pad if shorter than 30s
                    if len(audio) < 22050 * 30:
                        audio = np.pad(audio, (0, 22050*30 - len(audio)), mode='constant')
                    # Extract 40 MFCCs (better than 13)
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, hop_length=512)
                    X.append(mfccs.T)  # Shape: (~1292, 40)
                    y.append(genre)
                    valid_count += 1
                except Exception as e:
                    continue
    print(f"  {genre}: {valid_count} valid files")

print(f"\n✅ Total valid samples: {len(X)}")

# If you have fewer than 800 samples, something is wrong
if len(X) < 800:
    print("⚠️ Warning: Less than 800 valid files. Dataset may be corrupted.")
    print("   Continue anyway...")

X = np.array(X)
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

# Build robust model
model = models.Sequential([
    layers.Input(shape=(X.shape[1], X.shape[2])),
    layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.MaxPooling1D(4),
    
    layers.Conv1D(128, kernel_size=5, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.MaxPooling1D(4),
    
    layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.MaxPooling1D(4),
    
    layers.GlobalAveragePooling1D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🧠 Training model (50 epochs)...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_val, y_val),
    verbose=1
)

# Final evaluation
y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 FINAL TEST ACCURACY: {acc:.2%}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save results
model.save("genre_model.h5")
print("\n✅ Model saved as 'genre_model.h5'")



