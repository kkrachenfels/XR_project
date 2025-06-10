import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pose-based music generation")
    parser.add_argument("--pose-model", type=str, default='../models/best_natural_pose_model.pkl',
                        help="name of the saved pickled model")
    parser.add_argument("--mp-model", type=str, default="../pose_landmarker_full.task",
                        help="path of the mediapipe model to use (.task file)")
    parser.add_argument("--use-musicgen", action="store_true", 
                        help="Enable MusicGen output instead of MIDI")
    return parser.parse_args()

# Parse arguments FIRST, before other imports
args = vars(parse_arguments())
USE_MUSICGEN = args['use_musicgen']

print(f"Arguments parsed successfully: {args}")
print(f"USE_MUSICGEN = {USE_MUSICGEN}")

model_path = args['mp_model']
pose_model_path = args['pose_model']

import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import threading
import queue
import sounddevice as sd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Import music modules conditionally to avoid argument parser conflicts
if USE_MUSICGEN:
    from musicgen_thread import music_thread_v2 as music_thread_func
    print("Using MusicGen thread")
else:
    from midi_thread import music_thread_v2 as music_thread_func
    print("Using MIDI thread")

notes = ["A", "A#/Bb", "B", "C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab"]
BASE_KEY = [3, "major"] # 
BASE_TIME = 4 # 4 or 3 
BASE_TEMPO = 60

NUM_PEOPLE = 2

def calculate_average_velocity(curr_landmarks, prev_landmarks):
    if prev_landmarks is None:
        return 0

    total_dist = 0
    count = 0
    for i in range(min(len(curr_landmarks), len(prev_landmarks))):
        dx = curr_landmarks[i].x - prev_landmarks[i].x
        dy = curr_landmarks[i].y - prev_landmarks[i].y
        dz = curr_landmarks[i].z - prev_landmarks[i].z
        dist = (dx**2 + dy**2 + dz**2) ** 0.5
        total_dist += dist
        count += 1

    return total_dist / count if count > 0 else 0

def calculate_pairwise_distance(p1_landmarks, p2_landmarks):
    # Use midpoint of the torso for better spatial estimate
    keypoints = [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                 mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                 mp.solutions.pose.PoseLandmark.LEFT_HIP,
                 mp.solutions.pose.PoseLandmark.RIGHT_HIP]

    distances = []
    for kp in keypoints:
        l1 = p1_landmarks[kp]
        l2 = p2_landmarks[kp]
        dist = np.sqrt((l1.x - l2.x) ** 2 + (l1.y - l2.y) ** 2 + (l1.z - l2.z) ** 2)
        distances.append(dist)

    return np.mean(distances)  # average torso distance in 3D

def calculate_arm_lift_shift(pose_landmarks, max_shift=4):
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
    
def calculate_new_key(note, shift):
    note += shift
    while note < 0:
        note += 12
    while note > 11:
        note -= 12
    return note

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
# command_queue = queue.Queue()
# return_queue = queue.Queue()

# m_thread = threading.Thread(target=music_thread_func, args=(command_queue, return_queue))
# m_thread.start()
# initial_command = {
#     'key': BASE_KEY[0],
#     'progression': BASE_KEY[1], 
#     'tempo': 1.0
# }
# command_queue.put(initial_command)

frame_count = 0
last_progression = None
MAJOR_THRESHOLD = 1.3
MINOR_THRESHOLD = 1.2
last_sent_shift = 0
last_shift_time = 0
SHIFT_COOLDOWN = 1.0
last_tempo_factor = 1.0  # default tempo factor (1.0 = unchanged)
last_tempo_sent_time = 0
TEMPO_COOLDOWN = 1.0  # seconds between tempo changes
landmark_history = {}  # to track previous landmarks for each person


cur_key = BASE_KEY
cur_tempo = BASE_TEMPO
cur_time = BASE_TIME

current_music_command = {
    'key': cur_key[0],
    'progression': cur_key[1],
    'tempo': 1.0,
    'time': cur_time
}

command_queue = queue.Queue()
return_queue = queue.Queue()
m_thread = threading.Thread(target=music_thread_func, args=(command_queue, return_queue))
m_thread.start()
command_queue.put(current_music_command.copy())

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Webcam read failed.")
            break

        # === Feedback handling ===
        try:
            while True:
                feedback = return_queue.get_nowait()
                print(f"===> Got feedback: {feedback}")
                if 'audio_chunk' in feedback:
                    sd.play(feedback['audio_chunk'], feedback['sampling_rate'])
                if 'tempo' in feedback:
                    cur_tempo = feedback['tempo']
                if 'progression' in feedback:
                    cur_key[1] = feedback['progression']
                    current_music_command['progression'] = feedback['progression']
                if 'time' in feedback:
                    cur_time = feedback['time']
                    current_music_command['time'] = feedback['time']
                if 'shift' in feedback:
                    cur_key[0] = calculate_new_key(cur_key[0], feedback['shift'])
                    current_music_command['key'] = cur_key[0]
        except queue.Empty:
            pass

        # === Pose processing ===
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # === Time signature detection ===
        if len(results.pose_landmarks) >= 2:
            dist = calculate_pairwise_distance(results.pose_landmarks[0], results.pose_landmarks[1])
            current_music_command['time'] = 4 if dist < 0.2 else 3
            command_queue.put(current_music_command.copy())
            print(f"3D Distance: {dist:.3f} → Time Signature: {current_music_command['time']}/4")

        # === Expression detection ===
        if results.pose_landmarks:
            openness_scores = [calculate_arm_openness(p) for p in results.pose_landmarks if calculate_arm_openness(p) is not None]
            if openness_scores:
                avg_open = np.mean(openness_scores)
                if (last_progression == 'major' and avg_open < MINOR_THRESHOLD) or \
                   (last_progression == 'minor' and avg_open > MAJOR_THRESHOLD) or last_progression is None:
                    new_prog = 'minor' if avg_open < 1.0 else 'major'
                    last_progression = new_prog
                    current_music_command['progression'] = new_prog
                    command_queue.put(current_music_command.copy())
                    print(f"Openness: {avg_open:.2f} → {new_prog}")

            # === Arm lift → Key shift ===
            avg_shift = int(np.round(np.mean([calculate_arm_lift_shift(p) for p in results.pose_landmarks])))
            now = time.time()
            if avg_shift != last_sent_shift and now - last_shift_time > SHIFT_COOLDOWN:
                delta = avg_shift - last_sent_shift
                last_sent_shift = avg_shift
                last_shift_time = now
                cur_key[0] = calculate_new_key(cur_key[0], delta)
                current_music_command['key'] = cur_key[0]
                command_queue.put(current_music_command.copy())
                print(f"Vertical arm delta → Shift: {delta:+}")

            # === Velocity → Tempo ===
            avg_velocities = []
            for i, pose_landmarks in enumerate(results.pose_landmarks):
                prev = landmark_history.get(i)
                velocity = calculate_average_velocity(pose_landmarks, prev)
                avg_velocities.append(velocity)
                landmark_history[i] = pose_landmarks

            if avg_velocities:
                avg_velocity = np.mean(avg_velocities)
                clamped = np.clip(avg_velocity, 0.003, 0.5)
                norm = (clamped - 0.003) / (0.5 - 0.003)
                tempo_factor = 1.5 + (1 - norm) * (0.5 - 1.5)
                #print(f"[DEBUG] avg_velocity={avg_velocity}, clamped={clamped}, tempo_factor={tempo_factor}")

                if abs(tempo_factor - last_tempo_factor) > 0.001 and now - last_tempo_sent_time > TEMPO_COOLDOWN:
                    current_music_command['tempo'] = tempo_factor
                    last_tempo_factor = tempo_factor
                    last_tempo_sent_time = now
                    command_queue.put(current_music_command.copy())
                    cur_tempo = int(60 * tempo_factor)  # ← this line ensures display is updated

                    print(f"Avg velocity: {avg_velocity:.4f} → Tempo factor: {tempo_factor:.2f}")

        # === Display ===
        annotated = draw_landmarks(rgb, results) if results.pose_landmarks else rgb.copy()
        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (650, 200), (0, 0, 0), -1)
        info_strings = [
            f"Tempo: {cur_tempo:.0f}",
            f"Time: {cur_time}/4",
            f"Key: {notes[cur_key[0]]} {cur_key[1]}"
        ]
        for i, text in enumerate(info_strings):
            cv2.putText(bgr, text, (10, 60 + 60 * i), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)

        cv2.imshow("PoseLandmarker - Multi Person", bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    command_queue.put({'type': 'stop'})
    m_thread.join()

except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    command_queue.put({'type': 'stop'})
    m_thread.join()