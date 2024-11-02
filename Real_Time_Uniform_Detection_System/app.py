import streamlit as st
import torch
import cv2
import numpy as np
import pyttsx3
from uniform_model import UniformModel

# Load the model and TTS engine
model = UniformModel()
model.load_state_dict(torch.load('optimized_uniform_model.pth'))
model.eval()
tts_engine = pyttsx3.init()

# Streamlit interface
st.title("Real-Time Uniform Detection System")

# Initialize camera variable
camera = None
last_prediction = None  # Variable to track last prediction for TTS
message_to_speak = None  # Variable to store the message to be spoken

# Function to process frames and detect uniform
def detect_and_alert(frame):
    resized_frame = cv2.resize(frame, (128, 128))  # Resize for faster processing
    img_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    prediction = model.predict(img_tensor)

    # Set message and color based on prediction
    if prediction == "Uniform":
        message = "Student is allowed to class."
        color = (0, 255, 0)
    else:
        message = "Student is not allowed to class."
        color = (0, 0, 255)

    # Draw message on the frame
    cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame, message

# Layout for buttons on the left and camera feed on the right
col1, col2 = st.columns([1, 2])  # Two columns

with col1:
    st.header("Control Panel")
    # Camera control buttons
    if st.button("Start Camera", key="start_camera"):
        if camera is None:
            camera = cv2.VideoCapture(0)
            st.write("Camera is ON")

    if st.button("Stop Camera", key="stop_camera"):
        if camera is not None:
            camera.release()
            camera = None
            st.write("Camera is OFF")

with col2:
    st.header("Camera Feed")
    frame_placeholder = st.empty()  # Placeholder for displaying camera frame

# Main application loop for video feed
if camera is not None:
    while True:
        ret, frame = camera.read()
        if ret:
            processed_frame, message = detect_and_alert(frame)  # Process the frame for detection
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(processed_frame, channels="RGB")

            # Check if the message has changed for TTS
            if message != last_prediction:
                message_to_speak = message  # Update the message to be spoken
                last_prediction = message  # Update last prediction
        else:
            st.warning("Failed to capture frame from camera.")
            break

# Text-to-Speech handling
if message_to_speak is not None:
    tts_engine.say(message_to_speak)
    tts_engine.runAndWait()  # Speak the message outside of the camera loop

# Ensure the camera is released when the application ends
if camera is not None:
    camera.release()
