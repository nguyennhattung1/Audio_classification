<<<<<<< HEAD
import pandas as pd
filepath = "7061-6-0-0.wav"
path = "UrbanSound8K"
data_path = path + "/metadata/UrbanSound8K.csv"
# Read metadata file

df = pd.read_csv(data_path)

# Construct file path by concatenating fold and file name
df["relative_path"] = (
    "/fold" + df["fold"].astype(str) + "/" + df["slice_file_name"].astype(str)
)
print(df.head(10))
class_id = df.loc[df["slice_file_name"] == filepath]["classID"].values

print(class_id)
print(df.head(10))
=======
#keep this ver, loop recording, time runner, +-1sec
import sys
import pyaudio
import wave
import threading
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QComboBox, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import QTimer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class AudioRecorder(QMainWindow):
    def __init__(self):
        super().__init__()

        self.recording = False
        #self.closing = False  # Flag to indicate if window is closing
        self.audio = pyaudio.PyAudio()
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.start_time = 0
        self.stream = None
        self.initUI()

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
        """ self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate() """
        self.lines[0].set_data([], [])  # Clear plot data
        self.canvas.draw()  # Redraw canvas
        """    
        print("Stopping recording...")
        self.closing = False  # Reset closing flag
        print("Closing flag reset to False")"""

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
            # Plotting the live audio signal
            audio_data = np.frombuffer(data, dtype=np.int16)
            frequency = np.fft.fftfreq(len(audio_data), 1/self.RATE)
            magnitude = np.abs(np.fft.fft(audio_data))
            self.lines[0].set_data(frequency, magnitude)
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
            # Check if window is closing while recording
            '''
            if self.closing:
                print("collapsed")
                break
            '''
            
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
        return f'Recording: {hours:02d}:{minutes:02d}:{seconds:02d}'

    def closeEvent(self, event):
        # Ignore the close event and hide the window
        """ event.ignore()
        self.hide()
        self.closing = False """
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
>>>>>>> 02bb54be4b517cb20d5079f7af3668b458d8628a