# import cv2
# import numpy as np
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe.framework.formats import landmark_pb2

# # Model available to download here: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models
# model_path = "Downloads/pose_landmarker_full.task"

# video_source = 0

# num_poses = 2
# min_pose_detection_confidence = 0.5
# min_pose_presence_confidence = 0.5
# min_tracking_confidence = 0.5


# def draw_landmarks_on_image(rgb_image, detection_result):
#     pose_landmarks_list = detection_result.pose_landmarks
#     annotated_image = np.copy(rgb_image)

#     # Loop through the detected poses to visualize.
#     for idx in range(len(pose_landmarks_list)):
#         pose_landmarks = pose_landmarks_list[idx]

#         pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#         pose_landmarks_proto.landmark.extend([
#             landmark_pb2.NormalizedLandmark(
#                 x=landmark.x,
#                 y=landmark.y,
#                 z=landmark.z) for landmark in pose_landmarks
#         ])
#         mp.solutions.drawing_utils.draw_landmarks(
#             annotated_image,
#             pose_landmarks_proto,
#             mp.solutions.pose.POSE_CONNECTIONS,
#             mp.solutions.drawing_styles.get_default_pose_landmarks_style())
#     return annotated_image


# to_window = None
# last_timestamp_ms = 0


# def print_result(detection_result: vision.PoseLandmarkerResult, output_image: mp.Image,
#                  timestamp_ms: int):
#     global to_window
#     global last_timestamp_ms
#     if timestamp_ms < last_timestamp_ms:
#         return
#     last_timestamp_ms = timestamp_ms
#     # print("pose landmarker result: {}".format(detection_result))
#     to_window = cv2.cvtColor(
#         draw_landmarks_on_image(output_image.numpy_view(), detection_result), cv2.COLOR_RGB2BGR)


# base_options = python.BaseOptions(model_asset_path=model_path)
# options = vision.PoseLandmarkerOptions(
#     base_options=base_options,
#     running_mode=vision.RunningMode.LIVE_STREAM,
#     num_poses=num_poses,
#     min_pose_detection_confidence=min_pose_detection_confidence,
#     min_pose_presence_confidence=min_pose_presence_confidence,
#     min_tracking_confidence=min_tracking_confidence,
#     output_segmentation_masks=False,
#     result_callback=print_result
# )

# with vision.PoseLandmarker.create_from_options(options) as landmarker:
#     # Use OpenCV’s VideoCapture to start capturing from the webcam.
#     cap = cv2.VideoCapture(video_source)

#     # Create a loop to read the latest frame from the camera using VideoCapture#read()
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             print("Image capture failed.")
#             break

#         # Convert the frame received from OpenCV to a MediaPipe’s Image object.
#         mp_image = mp.Image(
#             image_format=mp.ImageFormat.SRGB,
#             data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
#         landmarker.detect_async(mp_image, timestamp_ms)

#         if to_window is not None:
#             cv2.imshow("MediaPipe Pose Landmark", to_window)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Path to downloaded model (.task file)
model_path = "./pose_landmarker_full.task"

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

def draw_landmarks(rgb_image, results):
    annotated = rgb_image.copy()
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
