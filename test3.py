import librosa
import matplotlib.pyplot as plt

import numpy as np
AUDIO_FILE = "7061-6-0-0.wav"
samples, sample_rate = librosa.load(AUDIO_FILE, sr=None, duration=4000)

# Check the number of channels
num_channels = samples.shape[0] if len(samples.shape) > 1 else 1

if num_channels == 1:
    print("The WAV file is mono.")
elif num_channels == 2:
    print("The WAV file is stereo.")
else:
    print(f"The WAV file has {num_channels} channels.")


plt.figure(figsize=(12, 4))
librosa.display.waveshow(samples, sr=sample_rate,color ="blue")
plt.title('Waveform')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()