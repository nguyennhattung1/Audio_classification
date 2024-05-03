#plot in pyQt
import queue
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import sounddevice as sd

device = 0
window = 1000
downsample = 1
channels = [1]
interval = 30

q = queue.Queue()
device_info = sd.query_devices(device, 'input')
samplerate = device_info['default_samplerate']
length = int(window * samplerate / (1000 * downsample))

plotdata = np.zeros((length, len(channels)))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Plot")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_title("Trini")
        self.lines = self.ax.plot(plotdata, color=(0, 1, 0.29))
        self.ax.set_facecolor((0, 0, 0))
        self.ax.set_yticks([0])
        self.ax.yaxis.grid(True)

        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        self.ani = FuncAnimation(self.fig, self.update_plot, interval=interval)

    def update_plot(self, frame):
        global plotdata
        while True:
            try:
                data = q.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            plotdata = np.roll(plotdata, -shift, axis=0)
            plotdata[-shift:, :] = data
        for column, line in enumerate(self.lines):
            line.set_ydata(plotdata[:, column])
        return self.lines

def audio_callback(indata, frames, time, status):
    q.put(indata[::downsample, [0]])

stream = sd.InputStream(device=device, channels=max(channels), samplerate=samplerate, callback=audio_callback)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()

    with stream:
        sys.exit(app.exec_())
