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

pygame.mixer.init()
pygame.mixer.music.set_volume(1.0)



def play_note(note_path):
    try:
        pygame.mixer.music.load(note_path)
        #my_sound = pygame.mixer.Sound(note_path)
        #my_sound.play()
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error playing {note_path}: {e}")



ap = argparse.ArgumentParser()

ap.add_argument("--pose-model", type=str, default='./models/best_pose_model.pkl',
                help="name of the saved pickled model")
ap.add_argument("--mp-model", type=str, default="./pose_landmarker_full.task",
                help="path of the mediapipe model to use (.task file)")
args = vars(ap.parse_args())

model_path = args['mp_model']
pose_model_path = args['pose_model']

# Set up video capture
cap = cv2.VideoCapture(0)  # or 1 for external cam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set up the pose landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=2,
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

POSE_TO_NOTE = {
    'A': './notes/A_fixed.wav',
    'B': './notes/B_fixed.wav',
    'C': './notes/C_fixed.wav',
    'D': './notes/D_fixed.wav',
    'E': './notes/E_fixed.wav',
    'F': './notes/F_fixed.wav',
    'G': './notes/G_fixed.wav',
    'A_sharp': './notes/A_sharp_fixed.wav',
    'B_flat': './notes/A_sharp_fixed.wav',
    'C_sharp': './notes/C_sharp_fixed.wav',
    'D_flat': './notes/C_sharp_fixed.wav',
    'D_sharp': './notes/D_sharp_fixed.wav',
    'E_flat': './notes/D_sharp_fixed.wav',
    'F_sharp': './notes/F_sharp_fixed.wav',
    'G_flat': './notes/F_sharp_fixed.wav',
    'G_sharp': './notes/G_sharp_fixed.wav',
    'A_flat': './notes/G_sharp_fixed.wav',
    'B_sharp': './notes/C_fixed.wav',
    'C_flat': './notes/B_fixed.wav',
    'E_sharp': './notes/F_fixed.wav',
    'F_flat': './notes/E_fixed.wav'
}

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
        for pose in cur_poses:
            if pose in POSE_TO_NOTE:
                sound_file = POSE_TO_NOTE[pose]
                print(f"before playing")
                if os.path.exists(sound_file) and not pygame.mixer.music.get_busy():
                    print(f"Playing {sound_file}")
                    play_note(sound_file)
                    time.sleep(1)
                else:
                    print(f"Sound file {sound_file} not found.")
        last_detected_poses = cur_poses

    # if set(cur_poses) != set(last_detected_poses):
    #     pass # play a changed note

    cv2.putText(bgr, f'Poses: {set(cur_poses)}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(bgr, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("PoseLandmarker - Multi Person", bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
