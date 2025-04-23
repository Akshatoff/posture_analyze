import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Setup MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Title
st.title("🧘‍♂️ Real-Time Posture Detection Web App")

# Streamlit camera input
stframe = st.empty()

# Initialize pose model
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Webcam Feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not available")
        break

    # Process image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    status = "Unable to detect posture"

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        def get_landmark(name):
            return landmarks[mp_pose.PoseLandmark[name].value]

        left_ear = get_landmark("LEFT_EAR")
        right_ear = get_landmark("RIGHT_EAR")
        left_shoulder = get_landmark("LEFT_SHOULDER")
        right_shoulder = get_landmark("RIGHT_SHOULDER")
        left_hip = get_landmark("LEFT_HIP")
        right_hip = get_landmark("RIGHT_HIP")

        slouch_threshold = 0.05
        lean_threshold = 0.04
        torso_threshold = 0.1

        slouching = (
            left_ear.y > left_shoulder.y + slouch_threshold and
            right_ear.y > right_shoulder.y + slouch_threshold
        )

        shoulder_diff = left_shoulder.y - right_shoulder.y
        leaning_left = shoulder_diff < -lean_threshold
        leaning_right = shoulder_diff > lean_threshold

        torso_misaligned = (
            abs(left_shoulder.x - left_hip.x) > torso_threshold or
            abs(right_shoulder.x - right_hip.x) > torso_threshold
        )

        if slouching:
            status = "⚠️ You're slouching!"
        elif leaning_left:
            status = "⚠️ Leaning left!"
        elif leaning_right:
            status = "⚠️ Leaning right!"
        elif torso_misaligned:
            status = "⚠️ Sit upright!"
        else:
            status = "✅ Perfect posture!"

    # Display status
    cv2.putText(image, status, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0) if "✅" in status else (0, 0, 255), 3)

    stframe.image(image, channels="BGR")

cap.release()
