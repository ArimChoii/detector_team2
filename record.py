#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyaudio
import wave
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def adjust_gain(data, gain):
    return data * gain

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 96000
CHUNK = 4096
RECORD_SECONDS = 5
OUTPUT_FILENAME = "test1.wav"
RESULT_FILENAME = "test1_result.wav"
USB_MIC_INDEX = 1
audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, input_device_index=USB_MIC_INDEX, frames_per_buffer=CHUNK)
print("녹음 시작...")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print('녹음 종료')

stream.stop_stream()
stream.close()
audio.terminate()

wf = wave.open(OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
print(f"파일 저장 완료: {OUTPUT_FILENAME}")

fs, data = wavfile.read(OUTPUT_FILENAME)

if len(data.shape) > 1:
    data = data[:, 0]

cutoff = 200.0
filtered_data = lowpass_filter(data, cutoff, fs, order=6)

gain = 5.0
amplified_data = adjust_gain(filtered_data, gain)
amplified_data = np.clip(amplified_data, -32768, 32767)
wavfile.write(RESULT_FILENAME, fs, np.int16(amplified_data))
print(f"Filtered and amplified audio saved as '{RESULT_FILENAME}'")
