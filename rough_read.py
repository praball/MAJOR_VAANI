import cv2
import numpy as np
import io

# Example file path
file_path = "data/चाय/0.jpg"

# Read the image file as binary data
with open(file_path, 'rb') as f:
    image_data = f.read()

# Convert binary data to numpy array
nparr = np.frombuffer(image_data, np.uint8)

# Decode the image using cv2.imdecode()
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Check if image reading was successful
if img is not None:
    # Process the image as needed
    print("Image shape:", img.shape)
else:
    print(f"Error: Unable to read image '{file_path}'")
