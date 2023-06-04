from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os.path

import pyaudio
from array import array
from struct import pack
import matplotlib.pyplot as plt

class VoiceVerifier:
    def __init__(self, model_file):
        self.model_file = model_file
        self.features = []
        self.labels = []
        self.model = None

    def save(self, username, sample_file):
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

    def verify(self, username, audio_file):
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
        features = self.extract_features(audio_file)
        if features is not None:
            if self.model is None:
                print("Model not trained. Please train the model first.")
                return False
            return self.model.predict([features])
        
    def predict_buffer(self, data):
        # Extract features from the signal
        features, _ = ShortTermFeatures.feature_extraction(data, SAMPLE_RATE, 0.050*SAMPLE_RATE, 0.025*SAMPLE_RATE)
        features = self.extract_features_buffer(data)
        if features is not None:
            if self.model is None:
                print("Model not trained. Please train the model first.")
                return False
            return self.model.predict([features])
        

    def train_model(self):
        if len(self.features) == 0 or len(self.labels) == 0:
            print("No voice information available for training.")
            return
        self.model = RandomForestClassifier()
        self.model.fit(self.features*5, self.labels*5)
        joblib.dump(self.model, self.model_file)
        print("Model trained and saved successfully.")

    def load_model(self):
        try:
            self.model = joblib.load(self.model_file)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("Model file not found. Train the model first.")

    def extract_features(self, audio_file):
        [sample_rate, signal] = audioBasicIO.read_audio_file(audio_file)
        if signal is not None:
            
            features, _ = ShortTermFeatures.feature_extraction(signal, sample_rate, 0.050*sample_rate, 0.025*sample_rate)
            # print(np.mean(features, axis=1))
            return np.mean(features, axis=1)
        else:
            return None
        
    def extract_features_buffer(self, signal):
        if signal is not None:
            features, _ = ShortTermFeatures.feature_extraction(signal, SAMPLE_RATE, 0.050*SAMPLE_RATE, 0.025*SAMPLE_RATE)
            # print(np.mean(features, axis=1))
            return np.mean(features, axis=1)
        else:
            return None




verifier = VoiceVerifier("voice_model.joblib")

# Save voice information for users
verifier.save("Sveta", r"records/sveta1.wav")
verifier.save("Sveta", r"records/sveta2.wav")
verifier.save("Sveta", r"records/sveta3.wav")
verifier.save("Sveta", r"records/sveta4.wav")
verifier.save("Sveta", r"records/sveta5.wav")
verifier.save("Roma", r"records/roma1.wav")
verifier.save("Roma", r"records/roma2.wav")
verifier.save("Roma", r"records/roma3.wav")
verifier.save("Roma", r"records/roma4.wav")
verifier.save("Roma", r"records/roma5.wav")
verifier.save("Silence", r"records/silence1.wav")
verifier.save("Silence", r"records/silence2.wav")
verifier.save("Silence", r"records/silence3.wav")
verifier.save("Silence", r"records/silence4.wav")
verifier.save("Silence", r"records/silence5.wav")

# Train the model
verifier.train_model()

# Load the trained model
verifier.load_model()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=1, rate=SAMPLE_RATE,
    input=True, output=True,
    frames_per_buffer=CHUNK_SIZE)
seq = 0
previus_len = 0
while True:

    r = array('h')
    for i in range(50):
        snd_data = array('h', stream.read(CHUNK_SIZE))
        r.extend(snd_data)

    # print(f'seq={seq} {verifier.predict_buffer(r)[0]}')
    # seq += 1
    print('\r' + ' '*previus_len, end='')
    predict = verifier.predict_buffer(r)[0]
    previus_len = len(predict)
    print('\r' + predict, end='')

'''
# Verify voice
result = verifier.verify("Sveta", r"records/roma3.wav")
if result:
    print("User verified.")
else:
    print("User not verified.")
'''
