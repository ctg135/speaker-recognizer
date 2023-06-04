import sys
import os
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures

'''

Utils for analysing signal

'''


if __name__ == '__main__':

    if len(sys.argv) == 1: 
        exit()
    file_name = sys.argv[1]

    if not os.path.isfile(file_name):
        print(f'{file_name} is not a file!')
    
    sample_rate, signal = audioBasicIO.read_audio_file(file_name)
    plt.plot(signal)
    
    plt.show()
    plt.title('Chromagram')
    ShortTermFeatures.chromagram(signal, sample_rate, 0.050*sample_rate, 0.025*sample_rate, plot=True)
    plt.title('Spectrogram')
    ShortTermFeatures.spectrogram(signal, sample_rate, 0.050*sample_rate, 0.025*sample_rate, plot=True)




