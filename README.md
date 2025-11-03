# Interactive ASL Learning System

Real-time, webcam-based **American Sign Language** learning + quiz app, with optional Arduino feedback.
- **final_assignment.py** – learn mode (watches the tutorial video timestamps, matches your hand).
- **quiz.py** – quiz mode (words built from A–Z using your configured letters).
- **fft_send.py** – (optional) tone/FFT helper for the Arduino message path.

## ✨ Features
- MediaPipe Hands tracking (no special gloves/sensors)
- Reference-timestamp matching against a tutorial video (A–Z)
- Quiz words configurable in code
- Optional Arduino serial output for hardware feedback (LCD/LED, buzzer, etc.)
- Simple `.env` support for local settings

## 📁 Structure
Interactive-ASL-Learning-System/
├── asl_tutorial.mp4 # (ignored by Git by default)
├── final_assignment.py # Learn mode
├── quiz.py # Quiz mode
├── fft_send.py # Optional FFT/tone utility
├── requirements.txt
├── .env # Local secrets/ports (ignored)
├── .gitignore
└── venv/ # Local virtual env (ignored)


## 🧪 Quick start (macOS)
```bash
# 0) In project root
python3 -m venv venv
source venv/bin/activate

# 1) Install deps
pip install -r requirements.txt

# 2) Create .env
cp .env.example .env   # if you create the example file below; otherwise create .env manually

.env template
# Serial port to your Arduino (ls /dev/cu.* to find it)
SERIAL_PORT=/dev/cu.usbmodem0000000000001
BAUD_RATE=9600

# (Optional) Mic device index for FFT path
MIC_DEVICE_INDEX=0

🔧 Configure

Inside the scripts, you’ll see configurable values near the top:

SERIAL_PORT / BAUD_RATE – or override via .env

VIDEO_PATH – path to asl_tutorial.mp4

REFERENCE_SIGNS_CONFIG – timestamps for A–Z (already filled based on your video)

QUIZ_WORDS – words to quiz (letters must exist in REFERENCE_SIGNS_CONFIG)

▶️ Run

Learn mode

python final_assignment.py


Quiz mode

python quiz.py


Optional FFT → Arduino

python fft_send.py

🧰 Troubleshooting

PyAudio build error on mac: brew install portaudio && pip install pyaudio

No camera permissions: allow Terminal/VS Code to access the camera in System Settings → Privacy & Security.

Serial not found: check ls /dev/cu.* and update SERIAL_PORT.

📄 License

MIT (add a LICENSE file if you want open-source)


---

# 5) (Optional) `.env.example`
Create this helpful template so others know what to set:

```ini
# Copy to .env and edit
SERIAL_PORT=/dev/cu.usbmodem201912341
BAUD_RATE=9600
MIC_DEVICE_INDEX=0