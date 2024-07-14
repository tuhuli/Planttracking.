import cv2
import numpy as np
from typing import List

# Initialize VideoCapture object
cap = cv2.VideoCapture("C:/SchoolApps/Bakalarka/Datasets/vineyard_videos/row_predice_videos/row_71_small_predikce.mp4")


# Get video properties (width, height, frames per second, etc.)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Check if the VideoCapture object was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Create VideoWriter object to save the modified video with Xvid codec
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("C:/SchoolApps/Bakalarka/Datasets/vineyard_videos/output_videos/colout.mp4", fourcc, fps, (width, height))


for i in range(207, 300):
    out_image = cv2.imread(f"../Datasets/vineyard_screenshots/row_SG19_small/original/original{i:04d}.png")
    if out_image is None:
        print(f"Error: Could not read image ../Datasets/vineyard_screenshots/row_SG19_small/original/original{i:04d}.png")
        continue

    print(out_image.shape[1])
    print(out_image.shape[0])
    print(width)
    print(height)

    if out_image.shape[1] != width or out_image.shape[0] != height:
        out_image = cv2.resize(out_image, (width, height))
    out.write(out_image)

cap.release()
out.release()
cv2.destroyAllWindows()