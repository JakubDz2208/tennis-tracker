from ultralytics import YOLO
import numpy as np
import cv2
import time
import torch

model_track = YOLO("runs/detect/train11/weights/best.pt")
model_seg = YOLO("runs/segment/train2/weights/last.pt")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_track.to(device)
model_seg.to(device)

def is_collision_detected(tracked_points):
    """
    Check if a collision is detected based on sudden motion change and height change.
    """

    if len(tracked_points) < 3:
        return False

    # Calculate the velocity vector between the last two tracked points
    last_point = tracked_points[-1][0]
    second_last_point = tracked_points[-2][0]
    velocity_vector = np.array(last_point) - np.array(second_last_point)

    # Calculate the magnitude of the velocity vector
    velocity_magnitude = np.linalg.norm(velocity_vector)

    # Set a velocity threshold
    velocity_threshold = 50  # Adjust as needed based on the speed of the ball

    # Check if the magnitude of the velocity vector exceeds the threshold
    if velocity_magnitude > velocity_threshold:
        # Calculate the height difference between the last two tracked points
        last_height = tracked_points[-1][0][1]
        second_last_height = tracked_points[-2][0][1]
        height_difference = last_height - second_last_height

        # Set a height threshold
        height_threshold = 50  # Adjust as needed based on the height difference indicating a collision

        # Check if the height difference exceeds the threshold
        if height_difference > height_threshold:
            return True

    return False

from collections import deque

def track_video_or_camera(video_source, device='cpu'):
    """
    Tracks points of interest in video file or live camera feed.
    
    Parameters:
    - video_source: Path to the video file or integer for camera index.
    - device: Device to run the segmentation and tracking models (default 'cpu').
    
    Press 'q' to exit the loop and close the window.
    """

    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    tracked_points = deque(maxlen=10)
    kernel = np.ones((100, 100), np.uint8)

    draw_circle = False
    circle_duration = 5.0
    circle_start_time = 0
    collision_position = None
    
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            segmentation_results = model_seg(frame, conf=0.5, device=device, show_labels=False, vid_stride=10)
            overlay = frame.copy()
            if segmentation_results[0].masks and segmentation_results[0].masks.shape[0] > 0:
                mask = segmentation_results[0].masks.xy[0]
                stencil = np.zeros(overlay.shape[:-1]).astype(np.uint8)
                cv2.fillPoly(stencil, [np.int32([mask])], 255)

                dilated_stencil = cv2.dilate(stencil, kernel, iterations=1)
                padding_mask = cv2.subtract(dilated_stencil, stencil)

                overlay[padding_mask == 255] = (0, 0, 255)

                overlay[stencil == 255] = (0, 255, 0)

                frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)


                tracking_results = model_track.track(frame, persist=True, conf=0.60, device=device)

                if tracking_results[0].boxes.shape[0] > 0:
                    first_box = tracking_results[0].boxes.data[0]

                    center_x = int((first_box[0] + first_box[2]) / 2)
                    center_y = int((first_box[1] + first_box[3]) / 2)

                    tracked_points.append(((center_x, center_y), time.time()))

                    if is_collision_detected(tracked_points):
                        draw_circle = True
                        circle_start_time = time.time()
                        collision_position = (center_x, center_y)

                current_time = time.time()
                tracked_points = [(pt, t) for (pt, t) in tracked_points if current_time - t < 1]

                if draw_circle and collision_position:
                    if time.time() - circle_start_time < circle_duration:
                        # Draw a circle on the collision position
                        cv2.circle(frame, collision_position, radius=5, color=(255, 0, 0), thickness=10)
                    else:
                        draw_circle = False


                for i in range(1, len(tracked_points)):
                    cv2.line(frame, tracked_points[i - 1][0], tracked_points[i][0], (0, 255, 0), 2)

                cv2.imshow("YOLOv8 Tracking", frame)
            else:
                cv2.imshow("YOLOv8 Tracking", frame)

            if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]:
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

video_path_or_camera_index = "tenis2.mp4"  # or 0 for webcam
track_video_or_camera(video_path_or_camera_index, device=device)