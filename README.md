# ✋ Interactive ASL Learning System (A–Z)

Real-time **ASL learning and quiz system** built using **MediaPipe Hands + Arduino**.
Learn sign language letters (A–Z), then get quizzed — with live feedback through LEDs/LCD on Arduino.

---

## 🎥 Demo

<p align="center">
  <b>🔹 Watch Full Demo (Google Drive)</b><br>
  <a href="https://drive.google.com/file/d/1WYSPlCFle2Q8LRD1QXZ60J2EaWAjOyiW/view?usp=sharing" target="_blank">
    ▶️ Click here to view full 2m25s demonstration
  </a>
</p>

<p align="center">
  <b>🔹 Play Demo Instantly (Compressed Preview)</b><br>
  <video src="assets/demo.mp4" width="720" controls autoplay muted playsinline loop></video>
</p>

---

## 📁 Project Structure

```
Interactive-ASL-Learning-System/
├── final_assignment.py       # Learn mode (A–Z gesture training)
├── quiz.py                   # Quiz mode (word quiz from gestures)
├── fft_send.py               # Optional Arduino tone/LED utility
├── requirements.txt
├── assets/
│   ├── demo.mp4              # Compressed preview video (~3.5 MB)
│   ├── Screen Recording 2025-07-26.mov   # Full recording (ignored, 1.0 GB)
│   └── screenshots/          # (Optional) app snapshots
├── .env                      # Local serial/mic config (not tracked)
├── .gitignore
└── venv/                     # Virtual environment (ignored)
```

---

## ⚙️ Quick Start (macOS)

```bash
# 1️⃣ Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 🧩 Requirements

```
opencv-python
mediapipe
numpy
pyserial
python-dotenv
pyaudio
```

> 💡 If PyAudio fails to install:
>
> ```bash
> brew install portaudio && pip install pyaudio
> ```

---

## 🔐 .env Configuration

Create a `.env` file in the project root:

```ini
# Serial port to your Arduino (run: ls /dev/cu.*)
SERIAL_PORT=/dev/cu.usbmodem201912341
BAUD_RATE=9600

# Optional: mic input index for FFT detection
MIC_DEVICE_INDEX=0
```

---

## 🧠 Key Parameters (in Python files)

| Variable                 | Description                                     |
| ------------------------ | ----------------------------------------------- |
| `VIDEO_PATH`             | Path to your ASL tutorial video                 |
| `REFERENCE_SIGNS_CONFIG` | A–Z timestamp references for gesture mapping    |
| `QUIZ_WORDS`             | Custom words for quiz mode                      |
| `MATCH_THRESHOLD`        | Euclidean distance tolerance (lower = stricter) |
| `QUIZ_DURATION_SEC`      | Duration for each sign in quiz mode             |
| `QUIZ_INTERVAL_SEC`      | Delay between quizzes                           |

---

## ▶️ How to Run

**1️⃣ Learn Mode**

```bash
python final_assignment.py
```

**2️⃣ Quiz Mode**

```bash
python quiz.py
```

**3️⃣ Arduino Tone/Signal Utility (Optional)**

```bash
python fft_send.py
```

---

## 🧩 Features

✅ MediaPipe Hands — no sensors or gloves needed
✅ Real-time ASL gesture recognition and evaluation
✅ Adjustable difficulty and timing thresholds
✅ Arduino integration for feedback LEDs/LCD
✅ `.env` file support for clean config management

---

## 🤪 Troubleshooting

| Issue                                | Solution                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------- |
| **No camera access**                 | macOS → System Settings → Privacy → Camera → Allow for Terminal / VS Code |
| **Serial not found**                 | Run `ls /dev/cu.*` and update `.env`                                      |
| **PyAudio error**                    | `brew install portaudio && pip install pyaudio`                           |
| **MediaPipe fails on Apple Silicon** | Try `pip install mediapipe==0.10.11`                                      |

---

## 📜 License

MIT License — freely usable for learning and research purposes.

---

## 🧮 Notes

The original `.mov` (1.05 GB) is stored on Google Drive for full playback.
The preview video was compressed using:

```bash
ffmpeg -i "Screen Recording 2025-07-26 at 17.29.39.mov" \
  -vf "scale=-2:720" -c:v libx264 -preset veryfast -crf 28 -an assets/demo.mp4
```

---

<p align="center">
  <sub>© 2025 Adwait Sanjay Varekar — Musashino University Data Science Dept.</sub><br>
  <sub>Interactive ASL Learning System | MediaPipe × Arduino × AI</sub>
</p>
