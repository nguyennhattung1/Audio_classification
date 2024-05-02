import queue
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

# Define audio variables
device = 0
window = 1000
downsample = 1
channels = [1]
interval = 30

# Create a queue
q = queue.Queue()
device_info = sd.query_devices(device, 'input')
samplerate = device_info['default_samplerate']
length = int(window * samplerate / (1000 * downsample))

# Initialize plotdata
plotdata = np.zeros((length, len(channels)))

# Create a Tkinter window
root = tk.Tk()
root.title("Audio Plot")

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("Trini")
lines = ax.plot(plotdata, color=(0, 1, 0.29))
ax.set_facecolor((0, 0, 0))
ax.set_yticks([0])
ax.yaxis.grid(True)

# Create a function to update the plot
def update_plot(frame):
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, column])
    return lines

# Create an audio callback function
def audio_callback(indata, frames, time, status):
    q.put(indata[::downsample, [0]])

# Create an audio stream
stream = sd.InputStream(device=device, channels=max(channels), samplerate=samplerate, callback=audio_callback)

# Create an animation
ani = FuncAnimation(fig, update_plot, interval=interval, blit=True)

# Embed the plot into Tkinter canvas
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Start the audio stream
with stream:
    root.mainloop()
