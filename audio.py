import os
import wave
import time 
import threading
import tkinter as tk
import pyaudio

class VoiceRecorder:
    def __init__(self):
        self.root = tk.Tk()
        self.root.resizable(False, False)
        self.button = tk.Button(text='🎙', font=("Arial", 120, "bold"), command=self.click_handler)
        self.button.pack()
        self.label = tk.Label(text='00:00:00')
        self.label.pack()
        self.recording = False
        self.chunk = 1024 * 2
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100  # in Hz
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.root.mainloop()
        
    def click_handler(self):
        if self.recording:
            self.recording = False
            self.button.config(fg="black")
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
        else:
            self.recording = True
            self.button.config(fg='red')
            threading.Thread(target=self.record).start()

    def record(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format, channels=self.channels, rate=self.rate,
                                      input=True, frames_per_buffer=self.chunk)
        self.frames = []
        start = time.time()
        while self.recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

            passed = time.time() - start
            secs = passed % 60
            mins = passed // 60
            hours = mins // 60
            self.label.config(text=f'{int(hours):02d}:{int(mins):02d}:{int(secs):02d}')

        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        exists = True
        i = 1
        while exists:
            if os.path.exists(f"recording{i}.wav"):
                i += 1
            else:
                exists = False

        with wave.open(f"recording{i}.wav", "wb") as sound_file:
            sound_file.setnchannels(self.channels)
            sound_file.setsampwidth(self.audio.get_sample_size(self.format))
            sound_file.setframerate(self.rate)
            sound_file.writeframes(b''.join(self.frames))

VoiceRecorder()
