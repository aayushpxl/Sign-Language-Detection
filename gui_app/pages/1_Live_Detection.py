import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

# ---------------- Streamlit Page Config ----------------
st.set_page_config(page_title="Live Detection", page_icon="📹", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.pred-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 30px;
    font-weight: 600;
    border: 2px solid #2563EB;
}
</style>
""", unsafe_allow_html=True)

st.title("📹 Live Sign Language Detection")
st.write("Use your webcam to detect signs in real time.")

# ---------------- Load Model ----------------
model = load_model("sign_language_model.h5")

# ---------------- Load actions ----------------
actions = np.array(sorted(os.listdir("MP_Data")))

# ---------------- Mediapipe Setup ----------------
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(img)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() \
                     if results.pose_landmarks else np.zeros(33 * 4)

    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() \
                   if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() \
                   if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])

# ---------------- Layout ----------------
col1, col2 = st.columns([3, 1])

with col2:
    st.write("### 🔮 Detected Sign")
    prediction_box = st.empty()
    prediction_box.markdown("<div class='pred-box'>Waiting...</div>", unsafe_allow_html=True)

with col1:
    st.write("### 🎥 Webcam Feed")
    frame_window = st.empty()

run = st.checkbox("Start Camera")

sequence = []
sentence = []
threshold = 0.7

# ---------------- Webcam Loop ----------------
if run:
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=False) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)

            # Mediapipe processing
            image, results = mediapipe_detection(frame, holistic)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            # Prediction when enough frames collected
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predicted_action = actions[np.argmax(res)]
                confidence = np.max(res)

                if confidence > threshold:
                    if len(sentence) == 0 or predicted_action != sentence[-1]:
                        sentence.append(predicted_action)

                    # Update right-side prediction box
                    prediction_box.markdown(
                        f"<div class='pred-box'>{predicted_action}</div>",
                        unsafe_allow_html=True
                    )

            # Show webcam
            frame_window.image(image, channels="BGR")

        cap.release()
