from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import os.path
from Feature import Feature

class SpeakerRecognizer:
    """
    Model that Recognize voice of speaker

    First need to `save()` samples with names and `train()` model, than use `predict()` or `verify()` to use
    """

    def __init__(self, model_file, feature: Feature):
        self.model_file = model_file
        self.features = []
        self.labels = []
        self.model = None
        self.window_size = 0.050
        self.step_size = 0.025
        self.feature: Feature = feature

    def save(self, username, sample_file):
        """
        Saves sample audio file and username
        """
        if not os.path.isfile(sample_file): 
            print(f'{sample_file} is not a file')
            return
        features = self.extract_features(sample_file)
        print(f'{sample_file} ', end='')
        if features is not None:
            self.features.append(features)
            self.labels.append(username)
            print(f" {username}")
        else:
            print(f"Failed to save voice information for user: {username}")

    def save_from_path(self, path):
        """
        Load samples from specified directory.
        It need a struct to work:
        ```
        path/
        --username1/
        ----sample_file
        ----...
        --username2/
        ----sample_file
        ----...
        ```
        """

        if not os.path.isdir(path): 
            print(f'{path} is not a directory')
            return
        sep = os.path.sep
        for (dirpath, dirnames, filenames) in os.walk(path):
            if dirpath.count(sep) > 1: continue
            if filenames == []: continue
            username = dirpath.split(sep)[1]
            counter = 0
            for file in filenames:
                counter = counter + 1
                sample_file = os.path.join(dirpath, file)
                features = self.extract_features(sample_file)
                if features is not None:
                    self.features.append(features)
                    self.labels.append(username)
                else:
                    print(f'Error {sample_file} {username}')
            print(f'Total {counter} samples for {username}')


    def verify(self, username, audio_file):
        """
        Verifies speaker with audio file
        """
        features = self.extract_features(audio_file)
        if features is not None:
            if self.model is None:
                print("Model not trained. Please train the model first.")
                return False
            predicted_label = self.model.predict([features])
            return predicted_label[0] == username
        else:
            print("Failed to extract features from the audio.")
            return False
    
    def predict(self, audio_file):
        """
        Predict value from a audio file
        """
        features = self.extract_features(audio_file)
        if features is not None:
            if self.model is None:
                print("Model not trained. Please train the model first.")
                return False
            return self.model.predict([features])[0], np.max(self.model.predict_proba([features]))
        
    def predict_buffer(self, signal, sample_rate):
        """
        Predict value from buffer
        """
        features = self.extract_features_buffer(signal, sample_rate)
        if features is not None:
            if self.model is None:
                print("Model not trained. Please train the model first.")
                return False
            return self.model.predict([features]), np.max(self.model.predict_proba([features]))
        
    def train_model(self):
        """
        Trains current model with saved features and labels for users
        """
        if len(self.features) == 0 or len(self.labels) == 0:
            print("No voice information available for training.")
            return
        self.model = RandomForestClassifier()
        self.model.fit(self.features*5, self.labels*5)
        joblib.dump(self.model, self.model_file)
        print("Model trained and saved successfully.")

    def load_model(self):
        """
        Load model with `joblib.load()` from saved file 
        """
        try:
            self.model = joblib.load(self.model_file)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("Model file not found. Train the model first.")
    
    def extract_features(self, audio_file):
        """
        Extract features from sound from audio file
        """
        return self.feature.extract_features(audio_file)
        
    def extract_features_buffer(self, buffer, sample_rate):
        """
        Extracts features from sound. Uses buffer, array of short int
        """
        return self.feature.extract_features_buffer(buffer, sample_rate)