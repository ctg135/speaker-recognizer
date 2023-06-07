from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from Feature import Feature
import numpy as np

class Raw(Feature):

    def __init__(self):
        self.window_size = 0.050
        self.step_size = 0.025
   
    def extract_features(self, audio_path):
        """
        Extract features from sound from audio file
        """
        [sample_rate, signal] = audioBasicIO.read_audio_file(audio_path)
        features, _ = ShortTermFeatures.feature_extraction(signal, sample_rate, self.window_size*sample_rate, self.step_size*sample_rate)
        return np.mean(features, axis=1)

    def extract_features_buffer(self, buffer, sample_rate):
        """
        Extracts features from sound. Uses buffer, array of short int
        """
        features, _ = ShortTermFeatures.feature_extraction(buffer, sample_rate, self.window_size*sample_rate, self.step_size*sample_rate)
        return np.mean(features, axis=1)