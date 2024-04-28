import codecs, os

import cv2
import numpy as np

# # f = codecs.open('hindi.txt', encoding='utf-8').read()

# # print (f)



# # Create a Hindi string
# hindi_text = "नमस्ते"

# # Print the Hindi string
# print(hindi_text)


# DATA_DIR = './data'
# for dir_ in os.listdir(DATA_DIR):
#     print("=== ",(os.path.join(DATA_DIR, dir_)))
#     b=0
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         if(b): break
#         b=1
#         actual_path = os.path.join(DATA_DIR, dir_, img_path)
#         print(actual_path)
#         # actual_path = actual_path.encode('utf-8')
#         img = cv2.imread(str(actual_path))


# hindi_string = "नमस्ते/here/there"

# # Encode the string to UTF-8
# utf8_encoded_string = hindi_string.encode('utf-8')

# # Print the UTF-8 encoded string
# print(utf8_encoded_string.decode())



actual_path = 'data/लाना/0.jpg'
with open(actual_path, 'rb') as f:
    image_data = f.read()
nparr = np.frombuffer(image_data, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
cv2.imshow('f',img)
while(1) :
    if cv2.waitKey(25) == ord('q'):break
# print(actual_path)
# img = cv2.imread(actual_path)