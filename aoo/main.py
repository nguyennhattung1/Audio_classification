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
        self.audio = pyaudio.PyAudio()
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.start_time = 0
        self.stream = None

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
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
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

    def update_timer(self):
        elapsed_time = time.time() - self.start_time
        self.label_timer.setText(self.format_time(elapsed_time))

    def format_time(self, elapsed_time):
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        return f' Recording: {hours:02d}:{minutes:02d}:{seconds:02d}'

    def closeEvent(self, event):
        if self.recording:
            event.ignore()
            self.hide()
        else:
            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    recorder = AudioRecorder()
    recorder.show()
    sys.exit(app.exec_())
