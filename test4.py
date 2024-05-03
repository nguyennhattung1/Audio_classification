#work without timer
import sys
import pyaudio
import wave
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QComboBox


class AudioRecorder(QMainWindow):
    def __init__(self):
        super().__init__()

        self.recording = False
        self.audio = pyaudio.PyAudio()
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100

        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 300, 150)
        self.setWindowTitle('Audio Recorder')

        self.device_combo = QComboBox(self)
        self.device_combo.setGeometry(50, 30, 200, 30)
        self.populate_input_devices()

        self.button = QPushButton('Start Recording', self)
        self.button.setGeometry(50, 80, 200, 30)
        self.button.clicked.connect(self.toggle_recording)

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
        if self.recording:
            self.recording = False
            self.button.setText('Start Recording')
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
        else:
            self.recording = True
            self.button.setText('Stop Recording')
            device_index = self.device_combo.currentData()
            threading.Thread(target=self.record, args=(device_index,)).start()

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
        self.save_recording(frames)

    def save_recording(self, frames):
        wf = wave.open("output.wav", "wb")
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b"".join(frames))
        wf.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    recorder = AudioRecorder()
    recorder.show()
    sys.exit(app.exec_())
