#  SignFusion: Real-Time Indian Sign Language Recognition Using Dual-Dataset Learning

SignFusion is a real-time **Indian Sign Language Recognition System** powered by **MediaPipe**, **OpenCV**, and **PyTorch**. This system learns from **3D motion capture data** and generalizes to real-world **2D webcam inputs** using a distance matrix–based embedding pipeline and a dual-model architecture built with **ResNet18 + ANN**.

---



## 🧠 Project Highlights

| Feature | Description |
|--------|-------------|
| 📊 Dual-Dataset Learning | Trained on **3D motion capture data**; inferred using **2D MediaPipe landmarks** |
| 🧍 Landmark Detection | Uses MediaPipe to extract hand, face, and shoulder landmarks (total 60 points) |
| 📐 Distance Matrix | Computes pairwise 3D distances → 1770 distances per frame → 1770×60 matrix |
| 🎨 Jet Color Mapping | Visualizes the distance matrix as a Jet-colored image for CNN input |
| 🧩 Model Architecture | Combines a **ResNet18 feature extractor** + **ANN classifier** |
| 📹 Real-time Prediction | Live webcam feed, smooth landmark padding, and high-speed inference |
| 🗃️ Dataset Pipeline | Clean scripts for data collection, transformation, visualization, and training |

---

## 🧠 Learning Strategy: Dual-Dataset with Hard Triplet Loss

This project uniquely leverages two different datasets:

| Dataset        | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| 🟩 **3D Motion Capture (MoCap)** | High-fidelity dataset used as the *anchor* for embedding learning |
| 🟦 **2D MediaPipe Landmarks**   | Used as *positive* and *negative* examples to train embeddings and for real-time inference |

### 🔀 Triplet Training Objective

Embeddings are trained using a **Hard Triplet Loss** strategy:

- **Anchor** = 3D MoCap sample (high quality)
- **Positive** = Corresponding 2D MediaPipe sample (same class)
- **Negative** = Random 2D sample (different class)

> This ensures generalization across dimensional shifts and improves intra-class similarity while maximizing inter-class separation.

### 💡 Model Flow

(3D Anchor) ——\
➝ EmbeddingResNet ➝ 512-dim Embeddings ➝ Triplet Loss
(2D Positive) —/
(2D Negative) —/



