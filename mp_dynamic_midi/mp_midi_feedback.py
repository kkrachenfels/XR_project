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
BASE_KEY = ["C", "Major"] # 
BASE_TIME = 4 # 4 or 3 
BASE_TEMPO = 120

NUM_PEOPLE = 2

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

def calculate_new_key(note, shift):
    note_int = ord(note.upper())
    note_int += shift
    if note_int < 65:
        note_int += 7
    if note_int > 71:
        note_int -= 7
    return chr(note_int)


ap = argparse.ArgumentParser()

ap.add_argument("--pose-model", type=str, default='../models/best_natural_pose_model.pkl',
                help="name of the saved pickled model")
ap.add_argument("--mp-model", type=str, default="../pose_landmarker_lite.task",
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
print("Press 'q' to quit.")
prev_time = time.time()


# Q for communication to MIDI music thread
command_queue = queue.Queue()
# Q for communication back to main thread
return_queue = queue.Queue()

# Start MIDI thread
m_thread = threading.Thread(target=music_thread_v2, args=(command_queue, return_queue))
m_thread.start()

collated_poses = []
last_mode_poses = []

frame_count = 0

cur_key = BASE_KEY
cur_tempo = BASE_TEMPO
cur_time = BASE_TIME

try: 
    while cap.isOpened():
        feedback = None
        success, frame = cap.read()
        if not success:
            print("Webcam read failed.")
            break

        # handle feedback from the midi thread
        try:
            feedback = return_queue.get_nowait()
            print(f"===>Got feedback: {feedback}===")

            if 'tempo' in feedback.keys():
                cur_tempo *= feedback['tempo']
            elif 'progression' in feedback.keys():
                cur_key[1] = feedback['progression']
            elif 'time' in feedback.keys():
                cur_time = feedback['time']
            elif 'shift' in feedback.keys():
                cur_key[0] = calculate_new_key(cur_key[0], feedback['shift'])
        except queue.Empty:
            pass
        # end feedback handling

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
        cur_poses += [0] * (NUM_PEOPLE - len(cur_poses))
        collated_poses.append(cur_poses)

        if frame_count % 30 == 0:
            collated_poses = np.array(collated_poses)
            mode_poses = []
            for column in collated_poses.T:
                m = mode(column)
                if m != '0': mode_poses.append(mode(column))

            print(mode_poses)
            for i, pose in enumerate(mode_poses):
                if pose in notes:
                    print(f"playing pose {pose}")
                    #play_note(pose)
                    send_command(pose)
            for i, pose in enumerate(last_detected_poses):
                if pose not in cur_poses:
                    #turn_off_note(pose)
                    pass
            last_mode_poses = mode_poses
            collated_poses = []

        cv2.putText(bgr, f'Poses: {last_mode_poses}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
        #cv2.putText
        info_string = f"Tempo: {cur_tempo}; Time: {cur_time}/4; Key: {cur_key[0]} {cur_key[1]}"
        cv2.putText(bgr, info_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)

        # Show FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        #cv2.putText(bgr, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 6)

        cv2.imshow("PoseLandmarker - Multi Person", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    send_command("stop")
    m_thread.join()

except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    send_command("stop")
    m_thread.join()

