import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pyAudioAnalysis import ShortTermFeatures
from pyAudioAnalysis import audioBasicIO


def delta(mfcc_frames):
    deltas     = (mfcc_frames[0:len(mfcc_frames)-2,:] - mfcc_frames[2:,:])/2
    new_frames = mfcc_frames[1:len(mfcc_frames) - 1,:]
    coeffs     = np.concatenate((new_frames, deltas),axis=1)
    return coeffs


audio_path = '../records/roma/sample1.wav'
x, sr = librosa.load(audio_path)
mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)

plt.plot(x)
plt.show()

plt.style.use('ggplot')


# mfcc
plt.subplot(3, 4, 1)
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.title('mfccs spectr')

plt.subplot(3, 4, 2)
plt.plot(mfccs)
plt.title('mfccs raw')

plt.subplot(3, 4, 3)
plt.plot(np.mean(mfccs, axis=0))
plt.title('mfccs axis=0')

plt.subplot(3, 4, 4)
plt.plot(np.mean(mfccs, axis=1))
plt.title('mfccs axis=1')

coef = librosa.feature.delta(mfccs)
# delta mfcc
plt.subplot(3, 4, 5)
librosa.display.specshow(coef, sr=sr, x_axis='time')
plt.title('delta mfcc spectr')

plt.subplot(3, 4, 6)
plt.plot(coef)
plt.title('delta mfcc raw')

plt.subplot(3, 4, 7)
plt.plot(np.mean(coef, axis=0))
plt.title('delta mfcc axis=0')

plt.subplot(3, 4, 8)
plt.plot(np.mean(coef, axis=1))
plt.title('delta mfcc axis=1')

# My features

[sample_rate, signal] = audioBasicIO.read_audio_file(audio_path)
features, names = ShortTermFeatures.feature_extraction(signal, sample_rate, 0.050*sample_rate, 0.025*sample_rate)
plt.subplot(3, 4, 10)
plt.plot(features)
plt.title('features')

plt.subplot(3, 4, 11)
plt.plot(np.mean(features, axis=0))
plt.title('features axis 0')

plt.subplot(3, 4, 12)
plt.plot(np.mean(features, axis=1))
plt.title('features axis 1')

plt.subplot(3, 4, 9)
plt.plot(signal)
plt.title('raw ')

plt.show()

# audio_path = '../samples/Sveta/sveta3.wav'
# x, sr = librosa.load(audio_path)
# mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
# coef = delta(mfccs)
# print(np.mean(mfccs, axis=0))
# plt.plot(np.mean(mfccs, axis=1))
# librosa.display.specshow(coef, sr=sr, x_axis='time')
