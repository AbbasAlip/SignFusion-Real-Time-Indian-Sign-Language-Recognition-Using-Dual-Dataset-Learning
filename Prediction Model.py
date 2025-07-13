# -*- coding: utf-8 -*-
"""
Sign Language Recognition - Inference Model
Updated on Mar 21, 2025
Features:
- Start/Stop recording functionality
- Real-time prediction of signs
- Distance vector approach with visualization
- Based on existing landmark extraction code
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
import os
import time
from datetime import datetime
import pandas as pd

# ----------------------
# Load Pre-trained Models
# ----------------------
EMBEDDING_MODEL_PATH = "C:/Users/pabba/OneDrive/Desktop/Updated Sign language projecct/embedding_model.pth"
CLASSIFIER_MODEL_PATH = "C:/Users/pabba/OneDrive/Desktop/Updated Sign language projecct/classifier_model.pth"
NUM_CLASSES = 73

# ----------------------
# Define Sign Labels
# ----------------------
SIGN_LABELS = {
    i: label for i, label in enumerate([
        "absent", "absorb", "accept", "accident", "bail", "balcony", "ball", "cabbage",
        "cake", "call", "calm", "decrease", "earth", "earthquake", "east", "eclipse",
        "farmer", "file", "film", "fine", "food", "games", "garlic", "geometry",
        "germany", "ghee", "glass", "goa", "goal", "gold", "haryana", "hindi",
        "history", "hockey", "ice cream", "idli", "inch", "india-1", "jail", "jailor",
        "jam", "karnataka", "kilogram", "kilometer", "laddu", "lawyer", "leader",
        "magnet", "malayalam", "mango", "national anthem", "nepal-1", "oath",
        "olympics", "orange", "page", "paisae", "pakistan", "paper", "questions",
        "quote", "race", "radio", "record", "science", "scooter", "tamil", "taxi",
        "urdu", "vapour", "vegetables", "vehicle", "west"
    ])
}

# ----------------------
# Define Models
# ----------------------
class EmbeddingResNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super(EmbeddingResNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(512, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)  

    def forward(self, x):
        features = self.resnet(x)
        features = torch.flatten(features, start_dim=1)  
        return self.bn(features)

class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=NUM_CLASSES):
        super(EmbeddingClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ----------------------
# MediaPipe Setup (From Original Code)
# ----------------------
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define Corrected Landmark Indices (from original script)
LEFT_EYE_LANDMARKS = [66, 70]
RIGHT_EYE_LANDMARKS = [336, 300]
FOREHEAD_LANDMARKS = [54, 109, 338, 284]
NOSE_LANDMARKS = [1, 0]
CHIN_LANDMARKS = [152, 18]
CHEEKBONE_LANDMARKS = [280, 50, 212, 432]

# Maximum Expected Landmark Counts
MAX_HANDS = 2
HAND_LANDMARKS_PER_HAND = 21
HAND_COORDS_PER_LANDMARK = 3
MAX_HAND_COORDS = MAX_HANDS * HAND_LANDMARKS_PER_HAND * HAND_COORDS_PER_LANDMARK  # 126

FACE_LANDMARKS = LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS + FOREHEAD_LANDMARKS + NOSE_LANDMARKS + CHIN_LANDMARKS + CHEEKBONE_LANDMARKS
EYE_COORDS_PER_LANDMARK = 3
MAX_FACE_COORDS = len(FACE_LANDMARKS) * EYE_COORDS_PER_LANDMARK  # 48

SHOULDER_LANDMARKS = [11, 12]  # Left and right shoulders
POSE_COORDS_PER_LANDMARK = 3
MAX_POSE_COORDS = len(SHOULDER_LANDMARKS) * POSE_COORDS_PER_LANDMARK  # 6

# ----------------------
# Load Trained Models
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directory for debugging
os.makedirs("debug_output", exist_ok=True)

# Load models if they exist, otherwise just continue for debugging
try:
    embedding_model = EmbeddingResNet(embedding_dim=512).to(device)
    embedding_model.load_state_dict(torch.load(EMBEDDING_MODEL_PATH, map_location=device))
    embedding_model.eval()

    classifier = EmbeddingClassifier(embedding_dim=512, num_classes=NUM_CLASSES).to(device)
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
    classifier.eval()
    models_loaded = True
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    print("Running in debug mode without prediction capability")
    models_loaded = False

# ----------------------
# Landmark Extraction Functions (From Original Code)
# ----------------------
def extract_hand_landmarks(results):
    """Extracts hand landmarks from MediaPipe results"""
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
    return landmarks

def extract_face_landmarks(results):
    """Extract face landmarks from MediaPipe results"""
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx in FACE_LANDMARKS:
                landmark = face_landmarks.landmark[idx]
                landmarks.extend([landmark.x, landmark.y, landmark.z])
    return landmarks

def extract_pose_landmarks(results):
    """Extract pose landmarks from MediaPipe results"""
    landmarks = []
    if results.pose_landmarks:
        for idx in SHOULDER_LANDMARKS:
            landmark = results.pose_landmarks.landmark[idx]
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    return landmarks

def pad_landmarks(landmarks, max_length):
    """Ensure exactly max_length values by padding or truncating"""
    return landmarks[:max_length] + [0.0] * max(0, max_length - len(landmarks))

# ----------------------
# Distance Computation and Visualization (From 2nd Script)
# ----------------------
def compute_distance_vector(landmarks_flat):
    """
    Compute pairwise distances between landmarks
    Reshape flat landmarks to 60x3 format first
    """
    try:
        # Reshape the flat landmarks into 60 landmarks with 3 coordinates each
        landmarks = np.array(landmarks_flat).reshape(60, 3)
        num_landmarks = len(landmarks)
        distances = []
        
        for i in range(num_landmarks):
            for j in range(i + 1, num_landmarks):  # Avoid duplicate distances
                dist = np.linalg.norm(landmarks[i] - landmarks[j])
                distances.append(dist)
        
        return np.array(distances)  # Should be 1770 distances
    except Exception as e:
        print(f"Error computing distance vector: {e}")
        return np.zeros(1770)  # Return zeros in case of error

def convert_to_distance_image(frames_data):
    """
    Convert a sequence of frames to a distance matrix image
    using the jet colormap.
    """
    try:
        # Process each frame to get distance vectors
        all_distances = []
        for frame_landmarks in frames_data:
            distance_vector = compute_distance_vector(frame_landmarks)
            all_distances.append(distance_vector)
        
        # Create distance matrix (1770 x num_frames)
        distance_matrix = np.array(all_distances).T
        
        # Generate a temporary file path
        temp_file = os.path.join("debug_output", f"distance_matrix_{int(time.time()*1000)}.jpg")
        
        # Create and save the distance matrix image
        fig, ax = plt.subplots(figsize=(6, 12), dpi=300)
        plt.imshow(distance_matrix, cmap='jet', interpolation='nearest', aspect='auto')
        ax.set_axis_off()
        fig.patch.set_alpha(0)
        plt.savefig(temp_file, bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        
        # Also save the raw data for debugging
        pd.DataFrame(distance_matrix).to_excel(temp_file.replace(".jpg", ".xlsx"), index=False)
        
        # Read the saved image and return it
        distance_image = cv2.imread(temp_file)
        return distance_image, temp_file
        
    except Exception as e:
        print(f"Error creating distance image: {e}")
        return None, None

# ----------------------
# Inference Function Using Trained Models
# ----------------------
def infer_sign(distance_image):
    """
    Perform sign prediction using the trained models
    """
    if not models_loaded or distance_image is None:
        return "Model Not Loaded", 0.0
    
    try:
        # Preprocess the distance image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # Convert to tensor and run through models
        image_tensor = transform(distance_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Extract features
            feature = embedding_model(image_tensor)
            # Classify features
            prediction = classifier(feature)
            probs = F.softmax(prediction, dim=1)
            confidence, sign_id = torch.max(probs, dim=1)
            
            sign_id = sign_id.item()
            confidence = confidence.item()
        
        return SIGN_LABELS.get(sign_id, "Unknown"), confidence
    
    except Exception as e:
        print(f"Error during inference: {e}")
        return "Error", 0.0

# ----------------------
# Save Recorded Frames (From Original Code)
# ----------------------
def save_landmarks(sign_name, frames_data):
    """Save landmarks to Excel file for future training"""
    # Create directory if it doesn't exist
    os.makedirs(sign_name, exist_ok=True)
    
    # Generate a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(sign_name, f"{sign_name}_coordinates_{timestamp}.xlsx")
    
    # Save to Excel file
    df = pd.DataFrame(frames_data)
    df.to_excel(file_path, index=False)
    print(f"Data saved at {file_path}")
    
    return file_path

# ----------------------
# Main Application with Start/Stop Controls
# ----------------------
def main():
    """Main application with start/stop controls and visualization"""
    # Initialize MediaPipe components
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=MAX_HANDS, min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
    pose = mp_pose.Pose(min_detection_confidence=0.5)
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Application states
    WAITING = 0    # Waiting for user to press start
    COUNTDOWN = 1  # Countdown before recording
    RECORDING = 2  # Actively recording frames
    PROCESSING = 3 # Processing frames and making prediction
    SAVING = 4     # Save data for training
    
    # Initial state
    state = WAITING
    frames_data = []
    frame_count = 0
    max_frames = 60
    countdown_time = 3
    countdown_start = 0
    
    # UI settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    
    # Result tracking
    last_prediction = "None"
    confidence = 0.0
    distance_image = None
    
    # Display settings
    show_landmarks = True
    show_distance_map = True
    save_mode = False
    
    print("Starting sign language recognition...")
    print("Press SPACE to start/stop recording")
    print("Press 'S' to toggle save mode")
    print("Press 'L' to toggle landmark visualization")
    print("Press 'D' to toggle distance map")
    print("Press 'Q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Create a copy for drawing
        display_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process landmarks
        hand_results = hands.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)
        pose_results = pose.process(frame_rgb)
        
        # Draw landmarks if enabled
        if show_landmarks:
            # Draw hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Draw face landmarks
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    for idx in FACE_LANDMARKS:
                        lm = face_landmarks.landmark[idx]
                        h, w, _ = display_frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(display_frame, (cx, cy), 2, (0, 255, 0), -1)
            
            # Draw pose landmarks (shoulders)
            if pose_results.pose_landmarks:
                for idx in SHOULDER_LANDMARKS:
                    lm = pose_results.pose_landmarks.landmark[idx]
                    h, w, _ = display_frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(display_frame, (cx, cy), 5, (255, 0, 0), -1)
        
        # Create instruction panel
        panel_width = 220
        instruction_panel = np.zeros((display_frame.shape[0], panel_width, 3), dtype=np.uint8)
        
        # Add title and instructions
        cv2.putText(instruction_panel, "Sign Language Recognition", (10, 30), font, 0.7, (255, 255, 255), 2)
        
        if save_mode:
            cv2.putText(instruction_panel, "MODE: SAVE DATA", (10, 55), font, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(instruction_panel, "MODE: PREDICT", (10, 55), font, 0.6, (0, 200, 255), 1)
        
        cv2.putText(instruction_panel, "Last Prediction:", (10, 85), font, 0.6, (200, 200, 200), 1)
        
        # Color based on confidence
        if confidence > 0.7:
            pred_color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.4:
            pred_color = (0, 255, 255)  # Yellow for medium confidence
        else:
            pred_color = (0, 0, 255)  # Red for low confidence
        
        cv2.putText(instruction_panel, last_prediction, (10, 115), font, 0.7, pred_color, 2)
        
        if confidence > 0:
            cv2.putText(instruction_panel, f"Confidence: {confidence:.2f}", (10, 145), font, 0.6, pred_color, 1)
        
        # Control instructions
        cv2.putText(instruction_panel, "Controls:", (10, 180), font, 0.6, (200, 200, 200), 1)
        cv2.putText(instruction_panel, "SPACE - Start/Stop", (10, 205), font, 0.5, (200, 200, 200), 1)
        cv2.putText(instruction_panel, "S - Toggle Save Mode", (10, 225), font, 0.5, (200, 200, 200), 1)
        cv2.putText(instruction_panel, "L - Toggle Landmarks", (10, 245), font, 0.5, (200, 200, 200), 1)
        cv2.putText(instruction_panel, "D - Distance Map", (10, 265), font, 0.5, (200, 200, 200), 1)
        cv2.putText(instruction_panel, "Q - Quit", (10, 285), font, 0.5, (200, 200, 200), 1)
        
        # State-specific UI updates
        if state == WAITING:
            cv2.putText(instruction_panel, "Status: Ready", (10, 320), font, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press SPACE to start", 
                       (display_frame.shape[1]//2 - 150, display_frame.shape[0]//2), 
                       font, 1, (0, 255, 0), 2)
            
        elif state == COUNTDOWN:
            # Calculate remaining time
            elapsed = time.time() - countdown_start
            remaining = max(0, countdown_time - int(elapsed))
            
            cv2.putText(instruction_panel, "Status: Get Ready!", (10, 320), font, 0.6, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Starting in {remaining}...", 
                       (display_frame.shape[1]//2 - 150, display_frame.shape[0]//2), 
                       font, 1.5, (0, 255, 255), 3)
            
            # Move to recording when countdown ends
            if elapsed >= countdown_time:
                state = RECORDING
                frame_count = 0
                frames_data = []
                
        elif state == RECORDING:
            # Extract and store landmarks
            hand_landmarks = extract_hand_landmarks(hand_results)
            face_landmarks = extract_face_landmarks(face_results)
            pose_landmarks = extract_pose_landmarks(pose_results)
            
            # Pad landmarks to fixed length
            hand_landmarks_padded = pad_landmarks(hand_landmarks, MAX_HAND_COORDS)
            face_landmarks_padded = pad_landmarks(face_landmarks, MAX_FACE_COORDS)
            pose_landmarks_padded = pad_landmarks(pose_landmarks, MAX_POSE_COORDS)
            
            # Combine all landmarks
            frame_landmarks = hand_landmarks_padded + face_landmarks_padded + pose_landmarks_padded
            
            # Verify landmark count (should be 180)
            if len(frame_landmarks) == 180:
                frames_data.append(frame_landmarks)
                frame_count += 1
            else:
                print(f"Warning: Incorrect landmark count: {len(frame_landmarks)} (expected 180)")
            
            # Draw progress bar
            progress = min(1.0, frame_count / max_frames)
            bar_width = int(display_frame.shape[1] * 0.7)
            bar_height = 30
            bar_x = (display_frame.shape[1] - bar_width) // 2
            bar_y = display_frame.shape[0] - 50
            
            # Background bar
            cv2.rectangle(display_frame, 
                         (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            
            # Progress bar
            filled_width = int(bar_width * progress)
            cv2.rectangle(display_frame, 
                         (bar_x, bar_y), 
                         (bar_x + filled_width, bar_y + bar_height), 
                         (0, 255, 0), -1)
            
            # Progress text
            cv2.putText(display_frame, f"RECORDING: {frame_count}/{max_frames} frames", 
                       (bar_x, bar_y - 10), font, 0.7, (0, 255, 0), 2)
            
            cv2.putText(instruction_panel, "Status: RECORDING", (10, 320), font, 0.6, (0, 0, 255), 2)
            
            # Move to processing when done
            if frame_count >= max_frames:
                state = PROCESSING
                
        elif state == PROCESSING:
            # Show processing status
            cv2.putText(instruction_panel, "Status: Processing", (10, 320), font, 0.6, (255, 165, 0), 2)
            cv2.putText(display_frame, "Processing sign...", 
                       (display_frame.shape[1]//2 - 150, display_frame.shape[0]//2), 
                       font, 1, (255, 165, 0), 2)
            
            # Show progress 
            combined_frame = np.hstack((display_frame, instruction_panel))
            cv2.imshow("Sign Language Recognition", combined_frame)
            cv2.waitKey(1)
            
            # Process the frames
            try:
                # Generate distance matrix image
                distance_image, temp_file = convert_to_distance_image(frames_data)
                
                if save_mode:
                    # Move to saving state if in save mode
                    state = SAVING
                else:
                    # Perform prediction
                    if models_loaded and distance_image is not None:
                        prediction, confidence = infer_sign(distance_image)
                        last_prediction = prediction
                    else:
                        last_prediction = "No model"
                        confidence = 0.0
                    
                    # Display result for 3 seconds
                    cv2.putText(display_frame, f"Sign: {last_prediction}", 
                               (display_frame.shape[1]//2 - 150, display_frame.shape[0]//2 - 40), 
                               font, 1.2, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Confidence: {confidence:.2f}", 
                               (display_frame.shape[1]//2 - 150, display_frame.shape[0]//2 + 40), 
                               font, 1, (0, 255, 0), 2)
                    
                    combined_frame = np.hstack((display_frame, instruction_panel))
                    cv2.imshow("Sign Language Recognition", combined_frame)
                    cv2.waitKey(3000)  # Show result for 3 seconds
                    
                    # Go back to waiting state
                    state = WAITING
                
            except Exception as e:
                print(f"Error during processing: {e}")
                state = WAITING
        
        elif state == SAVING:
            # Prompt for sign name
            sign_name = input("Enter sign name to save: ")
            if sign_name:
                # Save the data
                saved_file = save_landmarks(sign_name, frames_data)
                print(f"Saved sign data to {saved_file}")
                
                # Show confirmation
                cv2.putText(display_frame, f"Saved sign: {sign_name}", 
                           (display_frame.shape[1]//2 - 150, display_frame.shape[0]//2), 
                           font, 1.2, (0, 255, 0), 2)
                
                combined_frame = np.hstack((display_frame, instruction_panel))
                cv2.imshow("Sign Language Recognition", combined_frame)
                cv2.waitKey(2000)  # Show confirmation for 2 seconds
            
            # Return to waiting state
            state = WAITING
        
        # Add distance map to instruction panel if available
        if distance_image is not None and show_distance_map:
            # Resize to fit panel width
            resized_distance = cv2.resize(distance_image, (panel_width, 150))
            
            # Add to bottom of panel if there's room
            if instruction_panel.shape[0] > 350 + resized_distance.shape[0]:
                y_offset = 350
                instruction_panel[y_offset:y_offset+resized_distance.shape[0], :] = resized_distance
                cv2.putText(instruction_panel, "Distance Map:", (10, y_offset-5), font, 0.6, (200, 200, 200), 1)
        
        # Combine frame with instruction panel
        combined_frame = np.hstack((display_frame, instruction_panel))
        cv2.imshow("Sign Language Recognition", combined_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space bar - start/stop
            if state == WAITING:
                # Start recording
                state = COUNTDOWN
                countdown_start = time.time()
            elif state == RECORDING:
                # Stop current recording
                if frame_count > 10:  # Only process if we have enough frames
                    state = PROCESSING
                else:
                    # Not enough frames, go back to waiting
                    state = WAITING
        elif key == ord('s'):  # Toggle save mode
            save_mode = not save_mode
        elif key == ord('l'):  # Toggle landmarks
            show_landmarks = not show_landmarks
        elif key == ord('d'):  # Toggle distance map
            show_distance_map = not show_distance_map
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()