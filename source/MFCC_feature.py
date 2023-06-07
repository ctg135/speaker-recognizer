import librosa
import numpy as np
from Feature import Feature

class MFCC(Feature):

    def __init__(self):
        pass
   
    def extract_features(self, audio_path) -> np.ndarray: 
        """
        Extract MFCC features from sound from audio file
        """
        x, sr = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
        return np.mean(mfccs, axis=1)

    def extract_features_buffer(self, buffer, sample_rate) -> np.ndarray:
        """
        Extract MFCC from sound directly
        """
        mfccs = librosa.feature.mfcc(y=buffer, sr=sample_rate, n_mfcc=40)
        return np.mean(mfccs, axis=1)