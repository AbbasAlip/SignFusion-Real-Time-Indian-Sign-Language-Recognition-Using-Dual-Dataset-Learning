#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 16:23:37 2025

@author: pvvkishore
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import os

# ----------------------
# Updated Embedding Model with Batch Normalization and Flattening
# ----------------------
class EmbeddingResNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super(EmbeddingResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(512, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)  # Normalize embeddings
    
    def forward(self, x):
        features = self.resnet(x)
        features = torch.flatten(features, start_dim=1)  # Ensure 1D output
        return self.bn(features)

# ----------------------
# Hard Negative Mining Function (Fixed Dimension Issue)
# ----------------------
def get_hard_negative(anchor, positive, negatives):
    pos_dist = F.pairwise_distance(anchor, positive)
    negatives = negatives.view(negatives.shape[0], anchor.shape[1])  # Ensure correct dimension
    neg_distances = F.pairwise_distance(anchor, negatives)
    hardest_negative = negatives[torch.argmin(neg_distances)]
    return hardest_negative

# ----------------------
# Updated Triplet Loss using Cosine Similarity
# ----------------------
def triplet_loss(anchor, positive, negative, margin=0.3):
    pos_sim = F.cosine_similarity(anchor, positive)
    neg_sim = F.cosine_similarity(anchor, negative)
    loss = torch.mean(F.relu(neg_sim - pos_sim + margin))
    return loss

# ----------------------
# Save Model Function
# ----------------------
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")

# ----------------------
# Load Model Function
# ----------------------
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model

# ----------------------
# Training Function for Embedding Learning with Hard Triplet Loss
# ----------------------
def train_embedding_model(model, dataloader_kinect, dataloader_mediapipe, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for (anchor_images, _), (positive_images, _), (negative_images, _) in zip(dataloader_kinect, dataloader_mediapipe, dataloader_mediapipe):
            anchor_images, positive_images, negative_images = (
                anchor_images.to(device), positive_images.to(device), negative_images.to(device)
            )
            
            optimizer.zero_grad()
            anchor_embedding = model(anchor_images)
            positive_embedding = model(positive_images)
            negative_embedding = model(negative_images)
            
            # Ensure correct shape for negatives
            negative_embedding = negative_embedding.view(negative_embedding.shape[0], anchor_embedding.shape[1])
            hardest_negative = get_hard_negative(anchor_embedding, positive_embedding, negative_embedding)
            
            loss = triplet_loss(anchor_embedding, positive_embedding, hardest_negative)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        loss_history.append(total_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")
    
    save_model(model, "embedding_model.pth")
    
    return model

# ----------------------
# Train the Embedding Model with Kinect & MediaPipe Data
# ----------------------
# ----------------------
# Load Kinect & MediaPipe Datasets
# ----------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Normalize to fixed resolution
    transforms.ToTensor()
])

dataset_kinect = datasets.ImageFolder(root="MCM", transform=transform)
dataset_mediapipe = datasets.ImageFolder(root="MPM", transform=transform)

dataloader_kinect = DataLoader(dataset_kinect, batch_size=32, shuffle=True)
dataloader_mediapipe = DataLoader(dataset_mediapipe, batch_size=32, shuffle=True)

# ----------------------
# Train the Embedding Model with Kinect & MediaPipe Data
# ----------------------

embedding_model = EmbeddingResNet(embedding_dim=512)
trained_embedding_model = train_embedding_model(embedding_model, dataloader_kinect, dataloader_mediapipe)

# ----------------------
# Classification Model using Embeddings
# ----------------------
class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=10):
        super(EmbeddingClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

classifier = EmbeddingClassifier(embedding_dim=512, num_classes=len(dataset_mediapipe.classes)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# ----------------------
# Training the Classifier on MediaPipe Embeddings
# ----------------------
def train_classifier(embedding_model, classifier, dataloader_mediapipe, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model.to(device).eval()
    classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader_mediapipe:
            images, labels = images.to(device), labels.to(device)
            
            with torch.no_grad():
                embeddings = embedding_model(images)
            
            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    save_model(classifier, "classifier_model.pth")
    
    return classifier

trained_classifier = train_classifier(trained_embedding_model, classifier, dataloader_mediapipe)
