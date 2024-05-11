import sys
import pyaudio
import wave
import threading
import time
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QComboBox, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import QTimer
from PyQt5 import QtGui
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import torch
from preprocessing_data import AudioUtil
from torch import nn
from residual_block import Residual
from datetime import datetime

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
        self.model = self.load_model()  # Load model when application starts
        self.initUI()
        self.class_labels = {
    0: "air_conditioner", # tiếng máy lạnh
    1: "car_horn",# còi ô tô
    2: "children_playing",# âm thanh chơi đùa của trẻ em 
    3: "dog_bark",# tiếng chó sủa
    4: "drilling",# tiếng máy khoan
    5: "engine_idling",# âm thanh động cơ xe máy
    6: "gun_shot",# tiếng súng và tiếng lớn
    7: "jackhammer",# tiếng máy móc trong xây dựng hoặc thi công đường
    8: "siren",# còi báo động 
    9: "street_music"# âm nhạc đường phố
}
    def initUI(self):
        self.setWindowTitle("Audio Plot")
        self.setGeometry(100, 100, 1000, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Main window layout
        main_window_layout = QVBoxLayout()
        self.device_combo = QComboBox()
        self.populate_input_devices()
        main_window_layout.addWidget(self.device_combo)
        self.label_timer = QLabel(f'Recording: 00:00:00', self)
        main_window_layout.addWidget(self.label_timer)
        self.button = QPushButton('Start Recording', self)
        self.button.clicked.connect(self.toggle_recording)
        main_window_layout.addWidget(self.button)
        self.label_prediction = QLabel("", self)  # Label to display predicted class
        main_window_layout.addWidget(self.label_prediction)  # Add label for predicted class
        main_window_layout.addStretch(1)

        # Plot window layout
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_title("Audio Signal (FFT)")
        self.lines = self.ax.plot([], color=(0, 1, 0.29))
        self.ax.set_facecolor((0, 0, 0))
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_ylim(-32768, 32768)  # Adjust amplitude range as needed
        self.ax.yaxis.grid(True)
        self.canvas = FigureCanvas(self.fig)
        plot_layout.addWidget(self.canvas)

        main_layout.addLayout(main_window_layout)
        main_layout.addWidget(plot_widget)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        # Adjusting font sizes and styles for labels and buttons
        # Inside the initUI() method

        # Adjusting font sizes and styles for labels and buttons
        self.label_timer.setFont(QtGui.QFont("Arial", 16))  # Example font size and style
        self.label_prediction.setFont(QtGui.QFont("Arial", 16))  # Example font size and style
        self.button.setFont(QtGui.QFont("Arial", 14))  # Example font size and style

        # Increasing label size
        self.label_timer.setMinimumHeight(50)  # Adjust the height as needed
        self.label_prediction.setMinimumHeight(50)  # Adjust the height as needed

        # Styling the button
        self.button.setStyleSheet("background-color: #4CAF50; color: white; border: none; padding: 10px 24px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px;")

        # Setting background color for main window
        self.setStyleSheet("background-color: #f2f2f2;")
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
        device_index = self.device_combo.currentData()
        self.lines[0].set_data([], [])  # Clear plot data
        self.canvas.draw()

    def save_recording(self, frames):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"output.wav"
        wf = wave.open(filename, "wb")
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b"".join(frames))
        wf.close()
        return filename

    def update_timer(self):
        elapsed_time = time.time() - self.start_time
        self.label_timer.setText(self.format_time(elapsed_time))

    def format_time(self, elapsed_time):
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        return f'Recording: {hours:02d}:{minutes:02d}:{seconds:02d}'

    def closeEvent(self, event):
        if self.recording:
            event.ignore()
            self.hide()
        else:
            event.accept()

    def load_model(self):
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
            layers = []
            for i in range(num_residuals):
                if i == 0 and not first_block:
                    layers.append(Residual(num_channels, use_1x1conv=True, strides=2))
                else:
                    layers.append(Residual(num_channels))
            return layers

        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))

        net = nn.Sequential(b1, b2, b3, b4, b5,
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(512, 10))

        net.load_state_dict(torch.load("net.pt", map_location=torch.device('cpu')))
        net.eval()
        return net

    def predict_class(self, filepath):
        aud = AudioUtil.open(filepath)
        reaud = AudioUtil.resample(aud, 44100)
        rechan = AudioUtil.rechannel(reaud, 1)
        dur_aud = AudioUtil.pad_trunc(rechan, 4000)
        shift_aud = AudioUtil.time_shift(dur_aud, 0.4)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        input_tensor = torch.tensor(aug_sgram, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
        _, predicted_class = torch.max(output, 1)
        return predicted_class.item()

    def record(self, device_index):
        self.stream = self.audio.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      frames_per_buffer=self.CHUNK,
                                      input_device_index=device_index)
        frames = []
        interval_frames = []
        interval_duration = 4
        start_time = time.time()

        while self.recording:
            data = self.stream.read(self.CHUNK)
            frames.append(data)
            interval_frames.append(data)

            audio_data = np.frombuffer(data, dtype=np.int16)
            frequency = np.fft.fftfreq(len(audio_data), 1 / self.RATE)
            magnitude = np.abs(np.fft.fft(audio_data))
            self.lines[0].set_data(frequency, magnitude)
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()

            elapsed_time = time.time() - start_time
            if elapsed_time >= interval_duration:
                filename = self.save_recording(interval_frames)
                predicted_class = self.predict_class(filename)
                self.label_prediction.setText(f"Predicted class: {self.class_labels[predicted_class]}")
                print("Predicted class:", predicted_class)
                print("Predicted class: ", self.class_labels[predicted_class])
                interval_frames = []
                start_time = time.time()

        self.stream.stop_stream()
        self.stream.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    recorder = AudioRecorder()
    recorder.show()
    sys.exit(app.exec_())
