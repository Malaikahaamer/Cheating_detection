# app.py
import streamlit as st
import cv2
from utils import detect_cheating

st.set_page_config(page_title="AI Cheating Detection", layout="wide")

st.title("AI-Based Live Cheating Detection Web App")
st.markdown("**Live webcam monitoring using YOLOv8**")

run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

cap = None

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame, flags = detect_cheating(frame)

        FRAME_WINDOW.image(processed_frame)

        if flags:
            for flag in flags:
                st.error(f"⚠️ {flag}")

else:
    if cap:
        cap.release()
