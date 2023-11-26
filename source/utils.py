import numpy as np
import pyaudio
from SpeakerRecognizer import SpeakerRecognizer

MAX_INT16 = 2**15
FORMAT = pyaudio.paInt16
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024

def audio_buffer_to_float(chunk):
    '''
    Returns ndarray with dtype=np.float32
    '''
    decode_int = np.frombuffer(chunk, dtype=np.int16)
    decode_float = decode_int.astype(np.float32)
    audio_normalised = decode_float / MAX_INT16
    return audio_normalised

def predict_on_fly(recognizer: SpeakerRecognizer):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=SAMPLE_RATE,
        input=True, output=True,
    frames_per_buffer=CHUNK_SIZE)

    while True:
        r = []
        for _ in range(40):
            data = stream.read(CHUNK_SIZE)
            r.extend(audio_buffer_to_float(data))
        arr = np.asarray(r, dtype=np.float32)
        predict = recognizer.predict_buffer(arr, SAMPLE_RATE)[0]
        print(predict)

    p.close(stream)







