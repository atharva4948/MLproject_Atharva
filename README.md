# MLproject_Atharva

🎵 Music Genre Classification using Deep Learning

This project builds a robust deep learning model to classify music into different genres using raw audio files. The system processes .wav files, extracts meaningful audio features, and trains a Convolutional Neural Network (CNN) to accurately predict the genre of a given track.

🔍 Project Overview

The model uses the GTZAN dataset structure (genres_original/) where each genre contains multiple 30-second audio samples. Before training, the dataset is validated to filter out corrupted or unreadable audio files to ensure clean training data.

⚙️ Feature Engineering

Loaded audio using Librosa

Extracted 40 MFCC (Mel-Frequency Cepstral Coefficients) for better audio representation

Padded shorter audio clips to maintain consistent input shape

Converted labels using LabelEncoder

Used stratified train–validation–test split for balanced evaluation

🧠 Model Architecture

A 1D Convolutional Neural Network was designed with:

Multiple Conv1D layers

Batch Normalization for stable training

Dropout layers to reduce overfitting

Global Average Pooling

Dense layers with Softmax output

The model is trained for 50 epochs using the Adam optimizer and sparse categorical crossentropy loss.

📊 Evaluation

Final test accuracy is computed

Detailed classification report generated

Model saved as genre_model.h5 for future inference

🚀 Skills Demonstrated

Audio signal processing

Feature extraction using MFCC

Deep learning model design

Overfitting control techniques

Model evaluation and persistence

End-to-end ML pipeline implementatio
