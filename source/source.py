import pyaudio
from array import array
from SpeakerRecognizer import SpeakerRecognizer

recognizer = SpeakerRecognizer("voice_model.joblib")

# Save voice information for users
# recognizer.save_from_path(r'../samples')


recognizer.save("Sveta", r"../samples/Sveta/sveta1.wav")
recognizer.save("Sveta", r"../samples/Sveta/sveta2.wav")
recognizer.save("Sveta", r"../samples/Sveta/sveta3.wav")
recognizer.save("Sveta", r"../samples/Sveta/sveta4.wav")
recognizer.save("Roma", r"../samples/Roma/roma1.wav")
recognizer.save("Roma", r"../samples/Roma/roma2.wav")
recognizer.save("Roma", r"../samples/Roma/roma3.wav")
recognizer.save("Roma", r"../samples/Roma/roma4.wav")
recognizer.save("Silence", r"../samples/Silence/silence1.wav")
recognizer.save("Silence", r"../samples/Silence/silence2.wav")
recognizer.save("Silence", r"../samples/Silence/silence3.wav")
recognizer.save("Silence", r"../samples/Silence/silence4.wav")


# Train the model
recognizer.train_model()

# Load the trained model
recognizer.load_model()

recognizer.predict()

exit(r"../samples/Roma/roma5.wav")


def predict_on_fly(recognizer):
    FORMAT = pyaudio.paInt16
    SAMPLE_RATE = 44100
    CHUNK_SIZE = 1024

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=SAMPLE_RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)
    
    previous_len = 0
    while True:

        r = array('h')
        for i in range(50):
            snd_data = array('h', stream.read(CHUNK_SIZE))
            r.extend(snd_data)

        print('\r' + ' '*previous_len, end='')
        predict = recognizer.predict_buffer(r, SAMPLE_RATE)[0]
        previous_len = len(predict)
        print('\r' + predict, end='')


def verify_voice(recognizer):
# Verify voice
    result = recognizer.verify("Sveta", r"records/roma3.wav")
    if result:
        print("User verified.")
    else:
        print("User not verified.")

predict_on_fly(recognizer)