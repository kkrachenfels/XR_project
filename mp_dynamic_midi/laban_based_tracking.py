import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import argparse
import os
import threading
import queue
from statistics import mode
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from midi_thread import *

notes = ["A", "B", "C", "D", "E", "F", "G"]
NUM_PEOPLE = 2

def calculate_arm_lift_shift(pose_landmarks, max_shift=6):
    try:
        # Use average height of wrists compared to shoulders
        left_shoulder_y = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder_y = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y
        left_wrist_y = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y
        right_wrist_y = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].y

        shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        wrist_y = (left_wrist_y + right_wrist_y) / 2

        # Negative if arms are raised
        delta = shoulder_y - wrist_y

        # Map range: roughly [-0.5, +0.5] → [-max_shift, +max_shift]
        shift = int(np.clip(delta * 20, -max_shift, max_shift)) # max so far is +/- 6
        return shift
    except:
        return 0


def calculate_arm_openness(pose_landmarks):
    # Only valid if both shoulders and wrists are detected
    try:
        left_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]

        # Distance between wrists (horizontal openness)
        wrist_span = abs(left_wrist.x - right_wrist.x)
        shoulder_span = abs(left_shoulder.x - right_shoulder.x)

        # Normalize by shoulder span (to account for body size)
        openness = wrist_span / shoulder_span
        return min(max(openness, 0.0), 2.0)  # clamp to [0, 2] just in case
    except:
        return None


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def send_command(note):
    if note == "A":
        command_queue.put({'progression': 'minor'})
    if note == "B":
        command_queue.put({'progression': 'major'}) 
    if note == "C":
        command_queue.put({'time': 3})
    if note == "D":
        command_queue.put({'time': 4})
    if note == "E":
        command_queue.put({'shift': 1})
    if note == "F":
        command_queue.put({'shift': -1})
    if note == "stop":
        command_queue.put({'type': 'stop'})


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
    num_poses=NUM_PEOPLE,
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
prev_time = time.time()

# Q for communication to MIDI music thread
command_queue = queue.Queue()
return_queue = queue.Queue()

m_thread = threading.Thread(target=music_thread_v2, args=(command_queue, return_queue))
m_thread.start()

frame_count = 0
last_progression = None
MAJOR_THRESHOLD = 1.1
MINOR_THRESHOLD = 0.9
last_sent_shift = 0
last_shift_time = 0
SHIFT_COOLDOWN = 1.0

try: 
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Webcam read failed.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if results.pose_landmarks:
            openness_scores = []
            for pose in results.pose_landmarks:
                openness = calculate_arm_openness(pose)
                if openness is not None:
                    openness_scores.append(openness)

            if openness_scores:
                avg_openness = np.mean(openness_scores)

                if last_progression == 'major' and avg_openness < MINOR_THRESHOLD:
                    command_queue.put({'progression': 'minor'})
                    print(f"Openness: {avg_openness:.2f} → minor")
                    last_progression = 'minor'
                elif last_progression == 'minor' and avg_openness > MAJOR_THRESHOLD:
                    command_queue.put({'progression': 'major'})
                    print(f"Openness: {avg_openness:.2f} → major")
                    last_progression = 'major'
                elif last_progression is None:
                    progression = 'minor' if avg_openness < 1.0 else 'major'
                    command_queue.put({'progression': progression})
                    print(f"Openness: {avg_openness:.2f} → {progression}")
                    last_progression = progression

            
            shifts = []
            for pose in results.pose_landmarks:
                shift = calculate_arm_lift_shift(pose)
                shifts.append(shift)

            if shifts:
                avg_shift = int(np.round(np.mean(shifts)))
                now = time.time()

                # Only send if shift changed AND cooldown has passed
                if avg_shift != last_sent_shift and (now - last_shift_time) > SHIFT_COOLDOWN:
                    command_queue.put({'shift': avg_shift - last_sent_shift})
                    print(f"Vertical arm delta → Shift: {avg_shift:+}")
                    last_sent_shift = avg_shift
                    last_shift_time = now

        if results.pose_landmarks:
            annotated = draw_landmarks(rgb, results)
        else:
            annotated = rgb.copy()

        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        cv2.putText(bgr, f'Progression: {last_progression}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(bgr, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 6)

        cv2.imshow("PoseLandmarker - Multi Person", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    command_queue.put({'type': 'stop'})
    m_thread.join()

except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    command_queue.put({'type': 'stop'})
    m_thread.join()