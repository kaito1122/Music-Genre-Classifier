import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pywt
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import audioread
from pydub import AudioSegment

# Function to extract Mel spectrogram features
def extract_mel_spectrogram(file_path, sr=22050, n_mels=128):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=sr)
        # Extract Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_flattened = mel_spectrogram_db.flatten()  # Flatten for use in ML models
    except (FileNotFoundError, librosa.util.exceptions.LibrosaError, audioread.NoBackendError):
        print(f"Audio file {file_path} not found or cannot be processed. Using CSV features instead.")
        mel_spectrogram_flattened = None
    return mel_spectrogram_flattened

# Splitting the dataset into training, validation, and test sets
def split_data(df, test_size=0.2, val_size=0.2):
    train, test = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)
    return train, test

# Function to extract spectrograms and wavelet features\n",
def extract_features(file_path, sr=22050, n_mels=128, wavelet='db1'):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=sr)
        # Extract Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram_flattened = mel_spectrogram_db.flatten()  # Flatten for use in ML models

        # Extract wavelet features
        coeffs = pywt.wavedec(y, wavelet, level=5)
        wavelet_features = np.concatenate([np.array(c).flatten() for c in coeffs])

        # Extract chroma features
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_features_flattened = chroma_stft.flatten()  # Flatten for use in ML models

        # Combine all features into a single feature vector
        combined_features = np.concatenate((mel_spectrogram_flattened, wavelet_features, chroma_features_flattened), axis=0)
    except (FileNotFoundError, librosa.util.exceptions.LibrosaError, audioread.NoBackendError):
        # If audio file is not found or cannot be loaded, use features from CSV instead
        print(f"Audio file {file_path} not found or cannot be processed. Using CSV features instead.")
        combined_features = None

    return combined_features

def process_dataset2(df, audio_directory):
    features = []
    labels = []
    for idx, row in df.iterrows():
        file_path = os.path.join(audio_directory, row['label'], row['filename'])
        feature_vector = extract_features(file_path)
        # If audio features cannot be extracted, use CSV features
        if feature_vector is None:
            feature_vector = row.drop(['label']).filter(regex='^(?!filename)').values.astype(np.float32)
        features.append(feature_vector)
        labels.append(row['label'])
    # Ensure all feature vectors have the same length by padding or truncating
    max_length = max(len(f) for f in features)
    features = np.array([np.pad(f, (0, max_length - len(f)), 'constant') if len(f) < max_length else f[:max_length] for f in features])
    return features, np.array(labels)

def processed(filepath):
    df_sec = pd.read_csv(filepath)

    # Standardizing dataset before splitting
    scaler = StandardScaler()
    df_sec_numeric = df_sec.select_dtypes(include=['float64', 'int64']).copy()
    df_sec[df_sec_numeric.columns] = scaler.fit_transform(df_sec_numeric)

    # Splitting the dataset
    train_df_sec, test_df_sec = split_data(df_sec)

    # Extract and process features for dataset
    audio_directory = 'data/GTZAN/genres_original'

    # Features Extraction
    train_features, train_labels = process_dataset2(train_df_sec, audio_directory)
    test_features, test_labels = process_dataset2(test_df_sec, audio_directory)

    return [train_features, train_labels, test_features, test_labels]