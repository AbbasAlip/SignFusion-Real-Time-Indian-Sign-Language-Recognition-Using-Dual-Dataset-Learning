import cv2
import mediapipe as mp
import pandas as pd
import os
from datetime import datetime

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define Corrected Landmark Indices
LEFT_EYE_LANDMARKS = [66, 70]
RIGHT_EYE_LANDMARKS = [336, 300]
FOREHEAD_LANDMARKS = [54, 109, 338, 284]
NOSE_LANDMARKS = [1, 0]
CHIN_LANDMARKS = [152, 18]
CHEEKBONE_LANDMARKS = [280, 50, 212, 432]

FACE_LANDMARKS = LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS + FOREHEAD_LANDMARKS + NOSE_LANDMARKS + CHIN_LANDMARKS + CHEEKBONE_LANDMARKS
SHOULDER_LANDMARKS = [11, 12]

# Constants for landmark counts
MAX_HANDS = 2
HAND_LANDMARKS_PER_HAND = 21
HAND_COORDS_PER_LANDMARK = 3
MAX_HAND_COORDS = MAX_HANDS * HAND_LANDMARKS_PER_HAND * HAND_COORDS_PER_LANDMARK  # 126
MAX_FACE_COORDS = len(FACE_LANDMARKS) * 3  # 48
MAX_POSE_COORDS = len(SHOULDER_LANDMARKS) * 3  # 6

def extract_landmarks(results, landmark_indices):
    landmarks = []
    if results and results.landmark:
        for idx in landmark_indices:
            if idx < len(results.landmark):
                landmark = results.landmark[idx]
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            else:
                landmarks.extend([None, None, None])
    return landmarks

def fill_missing_values(curr_values, prev_values):
    if not prev_values:
        return [0 if v is None else v for v in curr_values]  # Default to 0 for first frame
    return [pv if v is None else v for v, pv in zip(curr_values, prev_values)]

def save_landmarks(sign_name, frames_data):
    folder_path = f"./{sign_name}"
    os.makedirs(folder_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder_path, f"{sign_name}_coordinates_{timestamp}.xlsx")
    pd.DataFrame(frames_data).to_excel(file_path, index=False)
    print(f"Data saved at {file_path}")

def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=MAX_HANDS, min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.5)
    
    sign_name = input("Enter sign name: ")
    frames_data, prev_hand_landmarks, prev_face_landmarks, prev_pose_landmarks = [], [], [], []
    
    while cap.isOpened() and len(frames_data) < 60:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results, face_results, pose_results = hands.process(frame_rgb), face_mesh.process(frame_rgb), pose.process(frame_rgb)
        
        # Extract landmarks and fill missing values with previous frame's data
        hand_landmarks = extract_landmarks(hand_results.multi_hand_landmarks[0] if hand_results.multi_hand_landmarks else None, range(21)) * MAX_HANDS
        face_landmarks = extract_landmarks(face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None, FACE_LANDMARKS)
        pose_landmarks = extract_landmarks(pose_results.pose_landmarks if pose_results.pose_landmarks else None, SHOULDER_LANDMARKS)
        
        hand_landmarks = fill_missing_values(hand_landmarks, prev_hand_landmarks)
        face_landmarks = fill_missing_values(face_landmarks, prev_face_landmarks)
        pose_landmarks = fill_missing_values(pose_landmarks, prev_pose_landmarks)
        
        prev_hand_landmarks, prev_face_landmarks, prev_pose_landmarks = hand_landmarks, face_landmarks, pose_landmarks
        frame_landmarks = hand_landmarks + face_landmarks + pose_landmarks
        
        print(f"Frame {len(frames_data) + 1}: Hands({len(hand_landmarks)}), Face({len(face_landmarks)}), Pose({len(pose_landmarks)}), Total: {len(frame_landmarks)}")
        frames_data.append(frame_landmarks)
        
        # Draw landmarks on frame
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                for idx in FACE_LANDMARKS:
                    h, w, _ = frame.shape
                    x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Sign Language Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    save_landmarks(sign_name, frames_data)

if __name__ == "__main__":
    main()
