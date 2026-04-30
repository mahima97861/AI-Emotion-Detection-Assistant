import cv2
import numpy as np
import random
import time
import pyttsx3
import matplotlib.pyplot as plt

# ---------------- VOICE ENGINE ----------------
engine = pyttsx3.init()

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

# ---------------- EMOTION DATA ----------------
emotion_messages = {
    "Happy": ["You look great 😄", "Keep smiling 😊"],
    "Sad": ["Stay strong ❤️", "Everything will be okay"],
    "Angry": ["Relax 😌", "Take a deep breath"],
    "Surprise": ["Wow 😲", "Unexpected moment!"],
    "Neutral": ["Stay calm 😐", "Keep going 👍"]
}

emotion_map = {"Happy":1, "Sad":2, "Angry":3, "Surprise":4, "Neutral":5}
emotion_history = []

current_emotion = "Neutral"
message = random.choice(emotion_messages["Neutral"])

last_update_time = 0
last_voice_time = 0

update_interval = 2
voice_interval = 5

# ---------------- TIMER ----------------
start_time = time.time()
duration = 20   # total run time

# ---------------- GRAPH SETUP ----------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_title("Live Emotion Trend")
ax.set_xlabel("Time")
ax.set_ylabel("Emotion Level")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Stop after fixed duration
    if time.time() - start_time > duration:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # -------- EMOTION UPDATE --------
    if time.time() - last_update_time > update_interval:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            brightness = np.mean(face)
            size = w * h

            if brightness > 160:
                current_emotion = "Happy"
            elif brightness < 60:
                current_emotion = "Sad"
            elif size > 35000:
                current_emotion = "Surprise"
            elif 60 <= brightness <= 90:
                current_emotion = "Angry"
            else:
                current_emotion = "Neutral"

            message = random.choice(emotion_messages[current_emotion])

            emotion_history.append(emotion_map[current_emotion])

        last_update_time = time.time()

    # -------- VOICE --------
    if time.time() - last_voice_time > voice_interval:
        engine.say(current_emotion)
        engine.runAndWait()
        last_voice_time = time.time()

    # -------- DRAW FACE --------
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # 🔴 Emotion (RED)
    cv2.putText(frame, f'Emotion: {current_emotion}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # ⚫ Message (BLACK)
    cv2.putText(frame, message, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    cv2.imshow("Emotion AI Assistant", frame)

    # -------- LIVE GRAPH UPDATE --------
    if len(emotion_history) > 1:
        line.set_xdata(range(len(emotion_history)))
        line.set_ydata(emotion_history)

        ax.relim()
        ax.autoscale_view()

        plt.draw()
        plt.pause(0.01)

    # Exit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEAN EXIT ----------------
cap.release()
cv2.destroyAllWindows()

plt.ioff()
plt.show()
