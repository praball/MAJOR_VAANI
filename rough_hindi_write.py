import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# Load a font with Hindi characters
font_path = "Akshar_Unicode.ttf"  # Provide the path to your Hindi font file
font_size = 40
font = ImageFont.truetype(font_path, font_size)

# Load the image
image_path = "0.jpg"  # Provide the path to your image file
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to PIL format
pil_image = Image.fromarray(image_rgb)
draw = ImageDraw.Draw(pil_image)

# Write Hindi text on the image
text = "नमस्ते"  # Your Hindi text here
text_width, text_height = draw.textsize(text, font=font)
text_position = ((image.shape[1] - text_width) // 2, (image.shape[0] - text_height) // 2)
draw.text(text_position, text, font=font, fill=(255, 255, 255))

# Convert the PIL image back to OpenCV format
cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Display the image
cv2.imshow('Hindi Text on Image', cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
