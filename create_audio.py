import numpy as np
import wave
import struct

# Parameters
filename = "alert.wav"
duration = 0.5  # seconds
frequency = 1000.0  # Hz (high-pitched beep)
sample_rate = 44100.0

# Generate a sine wave (beep)
t = np.linspace(0, duration, int(sample_rate * duration))
audio_data = np.sin(2 * np.pi * frequency * t) * 32767

# Save as .wav file
with wave.open(filename, 'w') as wav_file:
    wav_file.setparams((1, 2, int(sample_rate), len(audio_data), 'NONE', 'not compressed'))
    for sample in audio_data:
        wav_file.writeframes(struct.pack('h', int(sample)))

print(f"File '{filename}' has been created in your project folder!")