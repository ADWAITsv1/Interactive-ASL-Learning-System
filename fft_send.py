import numpy as np
import pyaudio
import serial
import time
from dotenv import load_dotenv
import os

load_dotenv()

SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/cu.usbmodem201912341")
BAUD_RATE = 9600
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 96000

# --- IMPORTANT: TUNE THIS VALUE! ---
SILENCE_THRESHOLD = 100

# --- MORSE CODE DICTIONARY ---
MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '..-.', 'F': '.-..'
}
# Note: Morse for E and F have been updated to be more distinct.

def get_char_for_freq(freq):
    """
    Returns a character ('A'-'F') based on specific frequency ranges.
    Returns None if the frequency is outside the defined ranges.
    """
    if 16000 <= freq < 16100:
        return 'A'  # Red
    elif 16100 <= freq < 16200:
        return 'B'  # Orange
    elif 16200 <= freq < 16300:
        return 'C'  # Blue
    elif 16300 <= freq < 16400:
        return 'D'  # Green
    elif 16400 <= freq < 16500:
        return 'E'  # Red + Green
    elif 16500 <= freq < 16600:
        return 'F'  # Red + Orange
    else:
        return None

def detect_freq(data, rate):
    """Analyzes audio data to find the dominant frequency."""
    try:
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(fft))
        peak = np.argmax(np.abs(fft)[1:]) + 1
        freq = abs(freqs[peak] * rate)
        return round(freq)
    except (IndexError, ValueError):
        return 0

def get_volume(data):
    """Calculates the volume (Root Mean Square) of the audio data."""
    return np.sqrt(np.mean(data.astype(float)**2))

# Connect to Arduino
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE)
    print(f" Connected to Arduino on {SERIAL_PORT}")
    time.sleep(2)
except serial.SerialException as e:
    print(f" Error: Could not connect to Arduino on {SERIAL_PORT}.")
    exit()

# Setup mic
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print(f"🎙️ Live listening started...")
print(f"Listening for specific high-frequency ranges to convert to Morse Code.")

try:
    while True:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        volume = get_volume(data)
        
        if volume > SILENCE_THRESHOLD:
            freq = detect_freq(data, RATE)
            char = get_char_for_freq(freq)
            
            if char:
                morse_code = MORSE_CODE_DICT.get(char)
                # Send data in the format "A:.-" followed by a newline
                data_to_send = f"{char}:{morse_code}\n"
                print(f"Range Detected! Freq: {freq} Hz → Sending '{data_to_send.strip()}'")
                arduino.write(data_to_send.encode())
                time.sleep(1.5)
            else:
                print(f"Ignoring out-of-range frequency: {freq} Hz      ", end='\r')
        else:
            print(f"Silence... Volume: {int(volume):<5}      ", end='\r')

except KeyboardInterrupt:
    print("\n Stopped.")
finally:
    if 'stream' in locals() and stream.is_active():
        stream.stop_stream()
        stream.close()
    if 'p' in locals():
        p.terminate()
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()
