import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
import threading
from tensorflow.keras.models import load_model

# ---------------------------- SETUP ----------------------------
model = load_model("sign_language_model.h5")
actions = np.array(['hello', 'thanks', 'iloveyou'])
sequence = []
threshold = 0.7

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Create holistic ONCE
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

frame_global = None
prediction_global = "..."

# Extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh   = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh   = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    # FIX: Add face landmarks subset (11 points)
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark[:11]]).flatten()
    else:
        face = np.zeros(11*3)

    return np.concatenate([pose, lh, rh, face])


# ---------------------------- BACKGROUND THREAD ----------------------------
def camera_loop():
    global frame_global, prediction_global, sequence

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe processing
        results = holistic.process(rgb)

        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Predict
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            if np.max(res) > threshold:
                prediction_global = actions[np.argmax(res)].upper()

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        frame_global = frame  # save for UI thread

# ---------------------------- TKINTER UI ----------------------------
window = tk.Tk()
window.title("Sign Language Detection")
window.geometry("900x650")
window.configure(bg="#1e1e1e")

title_label = tk.Label(
    window, text="Sign Language Detection",
    font=("Segoe UI", 22, "bold"),
    bg="#1e1e1e", fg="white"
)
title_label.pack(pady=10)

video_label = tk.Label(window, bg="black")
video_label.pack()

pred_label = tk.Label(window, text="...", font=("Segoe UI", 26, "bold"),
                      bg="#2b2b2b", fg="#42f58d")
pred_label.pack(pady=20)

def update_ui():
    if frame_global is not None:
        img = Image.fromarray(cv2.cvtColor(frame_global, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        pred_label.config(text=prediction_global)

    window.after(10, update_ui)

# Start camera thread
threading.Thread(target=camera_loop, daemon=True).start()

update_ui()
window.mainloop()
