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
from datetime import datetime

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
        self.resnet = models.resnet18(weights=None)  # ⚠️ Updated to use `weights=None`
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
# Load Trained Models
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_model = EmbeddingResNet(embedding_dim=512).to(device)
embedding_model.load_state_dict(torch.load(EMBEDDING_MODEL_PATH, map_location=device))
embedding_model.eval()

classifier = EmbeddingClassifier(embedding_dim=512, num_classes=NUM_CLASSES).to(device)
classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
classifier.eval()

# ----------------------
# MediaPipe Setup
# ----------------------
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5)

# ----------------------
# Define Landmark Indices
# ----------------------
LEFT_EYE_LANDMARKS = [66, 70]
RIGHT_EYE_LANDMARKS = [336, 300]
FOREHEAD_LANDMARKS = [54, 109, 338, 284]
NOSE_LANDMARKS = [1, 0]
CHIN_LANDMARKS = [152, 18]
CHEEKBONE_LANDMARKS = [280, 50, 212, 432]
SHOULDER_LANDMARKS = [11, 12]

FACE_LANDMARKS = (
    LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS + 
    FOREHEAD_LANDMARKS + NOSE_LANDMARKS + 
    CHIN_LANDMARKS + CHEEKBONE_LANDMARKS
)

# ----------------------
# Landmark Extraction & Padding (Fixed)
# ----------------------
def extract_hand_landmarks(results):
    """Extracts hand landmarks."""
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def extract_face_landmarks(results):
    """Extracts face landmarks using specific indices."""
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx in FACE_LANDMARKS:
                lm = face_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def extract_pose_landmarks(results):
    """Extracts pose landmarks for shoulders."""
    landmarks = []
    if results.pose_landmarks:
        for idx in SHOULDER_LANDMARKS:
            lm = results.pose_landmarks.landmark[idx]
            landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

def pad_landmarks(landmarks, max_length):
    """Ensures landmarks are padded or truncated to a fixed size."""
    return landmarks[:max_length] + [0.0] * max(0, max_length - len(landmarks))

# ----------------------
# Compute Distance Map & Convert to Jet Color Map
# ----------------------
def compute_distance_vector(landmarks):
    num_landmarks = len(landmarks) // 3
    distances = []
    for i in range(num_landmarks):
        for j in range(i + 1, num_landmarks):
            dist = np.linalg.norm(np.array(landmarks[i*3:(i+1)*3]) - np.array(landmarks[j*3:(j+1)*3]))
            distances.append(dist)
    return np.array(distances)

def convert_to_jet(distance_matrix):
    fig, ax = plt.subplots(figsize=(6, 12), dpi=300)
    plt.imshow(distance_matrix, cmap='jet', interpolation='nearest', aspect='auto')
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    plt.savefig("temp_jet.jpg", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    return "temp_jet.jpg"

# ----------------------
# Inference Function
# ----------------------
def infer_sign(frames):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    distance_matrix = np.array([compute_distance_vector(frame) for frame in frames]).T
    jet_img_path = convert_to_jet(distance_matrix)
    jet_img = cv2.imread(jet_img_path)
    
    frame_tensor = transform(jet_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feature = embedding_model(frame_tensor)
    
    with torch.no_grad():
        prediction = classifier(feature)
        sign_id = torch.argmax(prediction, dim=1).item()
    
    return SIGN_LABELS.get(sign_id, "Unknown")

# ----------------------
# Real-time Capture & Inference
# ----------------------
cap = cv2.VideoCapture(0)
frames = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_landmarks = pad_landmarks(extract_hand_landmarks(hands.process(frame_rgb)), 126)
    face_landmarks = pad_landmarks(extract_face_landmarks(face_mesh.process(frame_rgb)), 48)
    pose_landmarks = pad_landmarks(extract_pose_landmarks(pose.process(frame_rgb)), 6)

    frames.append(hand_landmarks + face_landmarks + pose_landmarks)
    frame_count += 1

    if frame_count == 60:
        print(f"📝 Predicted Sign: {infer_sign(frames)}")
        frame_count = 0
        frames = []

    cv2.imshow("Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
