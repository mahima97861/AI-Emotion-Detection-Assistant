# 🎭 AI Emotion Detection Assistant

An AI-powered real-time Emotion Detection Assistant built using Python, OpenCV, NumPy, Matplotlib, and Text-to-Speech technology.

The application captures live webcam video, detects faces, estimates emotions using facial characteristics, provides motivational feedback messages, speaks detected emotions aloud, and visualizes emotion trends in real-time through a dynamic graph.

---

## Features

### Real-Time Face Detection

* Detects human faces using OpenCV Haar Cascade Classifier.
* Draws bounding boxes around detected faces.

### Emotion Estimation

The system predicts emotions using image characteristics such as:

* Brightness
* Face size
* Facial region analysis

Supported emotions:

* Happy 😄
* Sad 😢
* Angry 😠
* Surprise 😲
* Neutral 😐

### Voice Assistant

* Announces detected emotions using Text-to-Speech.
* Built with pyttsx3.

### Motivational Feedback

Displays encouraging messages based on detected emotion.

Examples:

* "Keep smiling 😊"
* "Stay strong ❤️"
* "Relax 😌"

### Live Emotion Analytics

* Tracks emotional states over time.
* Generates a real-time emotion trend graph using Matplotlib.

### Interactive Interface

* Webcam feed display
* Emotion label overlay
* Personalized message overlay
* Live analytics visualization

---

## Technology Stack

| Technology   | Purpose              |
| ------------ | -------------------- |
| Python       | Core Programming     |
| OpenCV       | Face Detection       |
| NumPy        | Numerical Processing |
| Matplotlib   | Data Visualization   |
| pyttsx3      | Text-to-Speech       |
| Haar Cascade | Face Recognition     |

---

## Project Workflow

1. Webcam captures live video.
2. Face detector identifies faces.
3. Facial features are analyzed.
4. Emotion is estimated.
5. Motivational message is displayed.
6. Voice assistant announces emotion.
7. Emotion history is stored.
8. Live graph visualizes emotional trends.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/emotion-ai-assistant.git

cd emotion-ai-assistant
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Project

```bash
python emotion_ai.py
```

---

## Project Structure

```text
Emotion-AI-Assistant/
│
├── emotion_ai.py
├── requirements.txt
├── README.md
├── LICENSE


---

## Future Improvements

* Deep Learning Emotion Detection
* CNN-based Facial Expression Recognition
* TensorFlow Integration
* Emotion Accuracy Enhancement
* Emotion Dataset Training
* User Dashboard
* Emotion Report Export (PDF/CSV)
* Multi-Person Emotion Detection

---

## Limitations

Current emotion classification is based on heuristic image properties and should not be considered medically or psychologically accurate.

This project is intended for educational and research purposes only.

---

## Author

Mahima Mishra

BCA Student | Python Developer | AI & Machine Learning Enthusiast

---

## Acknowledgements

* OpenCV Community
* Python Software Foundation
* NumPy Developers
* Matplotlib Team
* pyttsx3 Contributors

---

## Star the Repository ⭐

If you found this project useful, consider giving it a star on GitHub.
