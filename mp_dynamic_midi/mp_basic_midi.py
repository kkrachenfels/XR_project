import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import argparse
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import pygame
import pygame.midi

pygame.mixer.init()
pygame.mixer.music.set_volume(1.0)
pygame.mixer.set_num_channels(5)
pygame.midi.init()

outport = pygame.midi.Output(1)
notes = ["A", "B", "C", "D", "E", "F", "G"]
middle_A = 57
DEFAULT_VELOCITY = 80
note_dir = "../notes"

def play_note(note):
    note_number = ord(note) - ord('A') + middle_A
    try:
        outport.note_on(note_number, velocity=DEFAULT_VELOCITY, channel=1)
    except Exception as e:
        print(f"Error playing {note_number}: {e}")

def turn_off_note(note):
    note_number = ord(note) - ord('A') + middle_A
    try:
        outport.note_off(note_number, velocity=DEFAULT_VELOCITY, channel=1)
    except Exception as e:
        print(f"Error turning off {note_number}: {e}")   


ap = argparse.ArgumentParser()

ap.add_argument("--pose-model", type=str, default='../models/best_natural_pose_model.pkl',
                help="name of the saved pickled model")
ap.add_argument("--mp-model", type=str, default="../pose_landmarker_full.task",
                help="path of the mediapipe model to use (.task file)")
args = vars(ap.parse_args())

model_path = args['mp_model']
pose_model_path = args['pose_model']

# Set up video capture
cap = cv2.VideoCapture(0)  # or 1 for external cam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

# Set up the pose landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=5,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = vision.PoseLandmarker.create_from_options(options)


with open(f'{pose_model_path}', 'rb') as f:
    model = joblib.load(f)

last_detected_poses = [] #track the last pose for each person

def predict_classes(pose_landmarks):
    # for each person
    classes = []
    for pose_landmarks in results.pose_landmarks:
        pose_coordinates = []
        for lm in pose_landmarks:
            pose_coordinates += [lm.x, lm.y, lm.z]
        pose_coordinates = np.around(pose_coordinates, decimals=9).reshape(1,99)
        predicted_class = model.predict(pose_coordinates)[0]
        predicted_prob = model.predict_proba(pose_coordinates)[0]
        #print(f"{predicted_class}: {predicted_prob}")
        classes.append(predicted_class)
    return classes


def draw_landmarks(rgb_image, results):
    annotated = rgb_image.copy()

    # for each person
    for pose_landmarks in results.pose_landmarks:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated,
            proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated


# Main loop
print("Press 'q' to quit.")
prev_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Webcam read failed.")
        break

    # Convert to RGB and wrap in MediaPipe Image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Provide timestamp for VIDEO mode
    timestamp_ms = int(time.time() * 1000)
    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    # Draw and show
    annotated = draw_landmarks(rgb, results)
    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    cur_poses = predict_classes(results.pose_landmarks)
    print(set(cur_poses))

    
    if set(cur_poses) != set(last_detected_poses):
        for i, pose in enumerate(cur_poses):
            if pose in notes:
                print(f"before playing")
                play_note(pose)
        for i, pose in enumerate(last_detected_poses):
            if pose not in cur_poses:
                turn_off_note(pose)
        last_detected_poses = cur_poses



    cv2.putText(bgr, f'Poses: {set(cur_poses)}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)

    # Show FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(bgr, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 6)

    cv2.imshow("PoseLandmarker - Multi Person", bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
