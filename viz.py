import os
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Function to visualize matches
def visualize_matches(img1, img2, mkpts0, mkpts1, filename=None):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(np.concatenate([img1, img2], axis=1))
    ax.axis('off')

    for (pt0, pt1) in zip(mkpts0, mkpts1):
        pt0 = pt0.astype(np.int32)
        pt1 = pt1.astype(np.int32)
        pt1[0] += img1.shape[1]
        
        ax.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], 'ro-', linewidth=1, markersize=3)
    
    if filename:
        plt.savefig(filename, bbox_inches='tight')
   
    plt.show()
    plt.close()

# Function to load image from file
def load_image(img_path):
    return cv2.imread(img_path)

# Main script
matches_file_path = "features/matches.h5"
keypoints_file_path = "features/keypoints.h5"
image_dir = 'dirname'  # Adjust this to your images directory

with h5py.File(matches_file_path, 'r') as matches_f, h5py.File(keypoints_file_path, 'r') as keypoints_f:
    for img_filename in matches_f.keys():
        img_path1 = os.path.join(image_dir,f"{img_filename}.jpg" )
        img1 = load_image(img_path1)
        kpts1 = keypoints_f[img_filename][:]
        
        for matched_img_filename in matches_f[img_filename].keys():
            img_path2 = os.path.join(image_dir, f"{matched_img_filename}.jpg")
            img2 = load_image(img_path2)
            kpts2 = keypoints_f[matched_img_filename][:]
            
            matches = matches_f[img_filename][matched_img_filename][:]
            mkpts0 = kpts1[matches[:, 0].astype(int)]
            mkpts1 = kpts2[matches[:, 1].astype(int)]
            
            visualize_matches(img1, img2, mkpts0, mkpts1)

# Print a message indicating the completion of the visualization
print("Matches visualization completed.")
