# ✋ Interactive ASL Learning System (A–Z)

Real-time **ASL learning and quiz system** using **MediaPipe Hands + Arduino**.  
Learn sign language letters (A–Z), then get quizzed — with live feedback via LEDs or LCD on Arduino!

---

## 🎥 Demo

**Full demo (Google Drive)**  
▶️ [Watch full performance (2 min 25 s)](https://drive.google.com/file/d/1WYSPlCFle2Q8LRD1QXZ60J2EaWAjOyiW/view?usp=sharing)

**Compressed preview (in this repo)**  
<video src="assets/demo.mp4" controls playsinline muted width="720"></video>

---

## 📁 Structure

Interactive-ASL-Learning-System/
├─ final_assignment.py # Learn mode (guided ASL video tracking)
├─ quiz.py # Quiz mode (word quiz using hand detection)
├─ fft_send.py # Optional: Arduino tone/LED signal utility
├─ requirements.txt
├─ assets/
│ ├─ demo.mp4 # Compressed preview (~3.5 MB)
│ ├─ Screen Recording 2025-07-26.mov (ignored, 1 GB original)
│ └─ screenshots/ # (Optional) snapshots
├─ .env # Local serial/mic config (not tracked)
├─ .gitignore
└─ venv/ # Local virtual environment (ignored)

yaml
Copy code

---

## ⚙️ Installation (macOS)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
Requirements
Copy code
opencv-python
mediapipe
numpy
pyserial
python-dotenv
pyaudio
If pyaudio fails on macOS:
brew install portaudio && pip install pyaudio

🔐 .env Configuration
Create a file named .env in the project root:

ini
Copy code
# Serial port for Arduino (find yours with ls /dev/cu.*)
SERIAL_PORT=/dev/cu.usbmodem201912341
BAUD_RATE=9600

# Optional mic index for FFT-based sound detection
MIC_DEVICE_INDEX=0
🧠 Key Parameters
Inside final_assignment.py and quiz.py, you can adjust:

Parameter	Description
VIDEO_PATH	Path to your ASL tutorial video
REFERENCE_SIGNS_CONFIG	A–Z timestamps for reference matching
QUIZ_WORDS	List of words to quiz (letters must exist above)
MATCH_THRESHOLD	Euclidean distance for match tolerance
QUIZ_DURATION_SEC	Time allowed per sign
QUIZ_INTERVAL_SEC	Delay between quizzes

▶️ Run the System
1️⃣ Learn Mode

bash
Copy code
python final_assignment.py
2️⃣ Quiz Mode

bash
Copy code
python quiz.py
3️⃣ Optional Arduino Signal Mode

bash
Copy code
python fft_send.py
🧩 Features
✅ MediaPipe Hands tracking (no gloves/sensors)
✅ Real-time hand landmark comparison against tutorial timestamps
✅ Adjustable quiz timing + difficulty
✅ Optional Arduino serial output for LEDs/LCD feedback
✅ Modular .env configuration for serial/mic settings

🧪 Troubleshooting
No camera access → System Settings → Privacy → Camera → Allow for Terminal or VS Code

Serial not found → ls /dev/cu.* and update .env

PyAudio error → brew install portaudio && pip install pyaudio

MediaPipe on M-series Macs → try pip install mediapipe==0.10.11

📜 License
MIT License — freely usable for learning and educational purposes.

💡 Notes
The full 1 GB .mov file is stored externally to save repository space.
If you need to reproduce the conversion command:

bash
Copy code
ffmpeg -i "Screen Recording 2025-07-26 at 17.29.39.mov" \
  -vf "scale=-2:720" -c:v libx264 -preset veryfast -crf 28 -an assets/demo.mp4