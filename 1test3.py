<<<<<<< HEAD
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
=======
#work well with time counter - best ver
import sys
import pyaudio
import wave
import threading
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QComboBox, QLabel
from PyQt5.QtCore import QTimer


class AudioRecorder(QMainWindow):
    def __init__(self):
        super().__init__()

        self.recording = False
        self.closing = False  # Flag to indicate if window is closing
        self.audio = pyaudio.PyAudio()
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.start_time = 0

        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 300, 200)
        self.setWindowTitle('Audio Recorder')

        self.device_combo = QComboBox(self)
        self.device_combo.setGeometry(50, 30, 200, 30)
        self.populate_input_devices()

        self.label_timer = QLabel(f' Recording: 00:00:00', self)
        self.label_timer.setGeometry(50, 70, 200, 30)

        self.button = QPushButton('Start Recording', self)
        self.button.setGeometry(50, 110, 200, 30)
        self.button.clicked.connect(self.toggle_recording)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)

    def populate_input_devices(self):
        devices = self.get_input_devices()
        for device in devices:
            self.device_combo.addItem(device['name'], device['index'])

    def get_input_devices(self):
        devices = []
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                devices.append({'name': device_info['name'], 'index': device_info['index']})
        return devices

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        self.button.setText('Stop Recording')
        device_index = self.device_combo.currentData()
        self.start_time = time.time()
        self.timer.start(1000)  # Update timer every second
        threading.Thread(target=self.record, args=(device_index,)).start()

    def stop_recording(self):
        self.recording = False
        self.button.setText('Start Recording')
        self.timer.stop()  # Stop the timer
        elapsed_time = time.time() - self.start_time
        self.label_timer.setText(self.format_time(elapsed_time))
        device_index = self.device_combo.currentData()
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def record(self, device_index):
        self.stream = self.audio.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      frames_per_buffer=self.CHUNK,
                                      input_device_index=device_index)
        frames = []
        while self.recording:
            data = self.stream.read(self.CHUNK)
            frames.append(data)
            # Check if window is closing while recording
            if self.closing:
                break
        self.save_recording(frames)

    def save_recording(self, frames):
        wf = wave.open("output.wav", "wb")
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b"".join(frames))
        wf.close()

    def update_timer(self):
        elapsed_time = time.time() - self.start_time
        self.label_timer.setText(self.format_time(elapsed_time))

    def format_time(self, elapsed_time):
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        return f' Recording: {hours:02d}:{minutes:02d}:{seconds:02d}'

    def closeEvent(self, event):
        # Ignore the close event and hide the window if recording
        if self.recording:
            event.ignore()  # Ignore the close event
            self.hide()  # Hide the window instead
            self.closing = True  # Set closing flag
        else:
            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    recorder = AudioRecorder()
    recorder.show()
    sys.exit(app.exec_())
>>>>>>> 02bb54be4b517cb20d5079f7af3668b458d8628a
