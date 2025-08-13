"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
objposecompare-1.py Python Program Documentation and Preinstall steps.

This Python program compares two Taekwondo training videos played in parallel using machine learning libraries, highlighting critical body movements through Object detection and pose detection using MediaPipe stack. The enhanced videos help students learn Taekwondo steps and actions correctly.

Author: Sanjana Vidyasagar 
Date: 2025-01-08

Step-1: In Windows machine download Python version 3.11 from https://www.python.org/downloads/release/python-3110/ and add python.exe file location to Environment PATH.

Step-2: Create a new virtual environment by running the following command
	python -m venv <Name of the new Virtual Machine>

Step-3: Open Command Prompt and Activate Virtual Environment 
	<Current Directory where Virtual environment is installed>\Scripts\activate.bat

Step-4: Install the following Machine Learning softwares by running the following command
	pip install opencv-python
	pip install mediapipe

Step-5: Copy the Python file and Taekwando Training videos from GIT repo to Virtual Environment home directory 
	objposecompare-1.py
	video1-T.mp4
	video2-T.mp4

Step-6: Run the python program
<Current Directory where Virtual environment is installed>\python objposecompare-1.py
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe models
mp_objectron = mp.solutions.objectron
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load Objectron model (for 3D object detection)
objectron = mp_objectron.Objectron(static_image_mode=False, max_num_objects=5, min_detection_confidence=0.5, model_name='Chair')

# Load Pose model
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load two Taekwando training videos
video1_path = "video1-T.mp4"
video2_path = "video2-T.mp4"

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        break  # Stop if videos end

    # Resize frames to make sure they are the same size  
    frame1 = cv2.resize(frame1, (640, 480))
    frame2 = cv2.resize(frame2, (640, 480))

    # Convert frames to RGB for Mediapipe processing
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # Process Object Detection
    results_obj1 = objectron.process(frame1_rgb)
    results_obj2 = objectron.process(frame2_rgb)

    # Process Pose Detection
    results_pose1 = pose.process(frame1_rgb)
    results_pose2 = pose.process(frame2_rgb)

    # Draw Objectron results (3D object detection)
    if results_obj1.detected_objects:
        for obj in results_obj1.detected_objects:
            mp_drawing.draw_landmarks(frame1, obj.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
    
    if results_obj2.detected_objects:
        for obj in results_obj2.detected_objects:
            mp_drawing.draw_landmarks(frame2, obj.landmarks_2d, mp_objectron.BOX_CONNECTIONS)

    # Draw Pose results
    if results_pose1.pose_landmarks:
        mp_drawing.draw_landmarks(frame1, results_pose1.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    if results_pose2.pose_landmarks:
        mp_drawing.draw_landmarks(frame2, results_pose2.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Stack both frames side by side for comparison
    combined_frame = np.hstack((frame1, frame2))

    # Show the result
    cv2.imshow("Video Comparison (Object & Pose Detection)", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()
