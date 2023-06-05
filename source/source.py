import pyaudio
from array import array
from VoiceVerifier import VoiceVerifier

verifier = VoiceVerifier("voice_model.joblib")

# Save voice information for users
verifier.save_from_path(r'samples')

'''
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
'''

# Train the model
verifier.train_model()

# Load the trained model
verifier.load_model()

def predict_on_fly(verifier):
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

        # print(f'seq={seq} {verifier.predict_buffer(r)[0]}')
        # seq += 1
        print('\r' + ' '*previous_len, end='')
        predict = verifier.predict_buffer(r, SAMPLE_RATE)[0]
        previous_len = len(predict)
        print('\r' + predict, end='')


def verify_voice(verifier):
# Verify voice
    result = verifier.verify("Sveta", r"records/roma3.wav")
    if result:
        print("User verified.")
    else:
        print("User not verified.")

predict_on_fly(verifier)