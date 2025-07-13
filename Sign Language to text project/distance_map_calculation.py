import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_distance_vector(landmarks):
    """
    Computes the pairwise distances for a single frame's landmarks.
    Returns a 1D array of size 1326 (52 choose 2).
    """
    num_landmarks = len(landmarks)
    distances = []
    
    for i in range(num_landmarks):
        for j in range(i + 1, num_landmarks):  # Avoid duplicate distances
            dist = np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[j]))
            #dist = np.linalg.norm(np.array(landmarks[i*3:(i+1)*3]) - np.array(landmarks[j*3:(j+1)*3]))
            distances.append(dist)
    
    return np.array(distances)

def process_single_excel(file_path):
    print(f"Processing {file_path}...")
    df = pd.read_excel(file_path)

    num_frames = len(df)  # Should be 60 frames
    landmarks_per_frame = df.to_numpy().reshape(num_frames, 60, 3)  # Reshape into (frames, landmarks, xyz)

    all_distances = []

    for frame_idx in range(num_frames):
        frame_landmarks = landmarks_per_frame[frame_idx]  # Extract 60 landmarks for the frame
        distance_vector = compute_distance_vector(frame_landmarks)  # Get 1770 distances
        all_distances.append(distance_vector)

    distance_matrix = np.array(all_distances).T  # Shape (1770, 60)
    
    # Save Distance Matrix
    fig, ax = plt.subplots(figsize=(6, 12), dpi=300)
    new_file_path = file_path.replace("coordinates", "distance_matrix")
    pd.DataFrame(distance_matrix).to_excel(new_file_path, index=False)
    print(f"Distance matrix saved at {new_file_path}")

    # Visualize Distance Matrix
    plt.imshow(distance_matrix, cmap='jet', interpolation='nearest',aspect='auto')
    ax.set_axis_off()  # Remove axes
    fig.patch.set_alpha(0)  # Transparent background
    #plt.colorbar()
    #plt.title("Distance Matrix (1326 x 60)")
    image_path = new_file_path.replace(".xlsx", ".jpg")
    plt.savefig(image_path,bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    print(f"Distance matrix image saved at {image_path}")

if __name__ == "__main__":
    file_path = "C:/Users/pabba/OneDrive/Desktop/Updated Sign language projecct/ghee/ghee_coordinates_20250311_134920.xlsx" # Update your path
    process_single_excel(file_path)
