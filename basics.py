import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # if id == 4:
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

"""
from sklearn.model_selection import train_test_split  
A_train, A_test, B_train, B_test = train_test_split(mnist.data,mnist.target, test_size=0.2, random_state=45)  
B_train  = B_train.astype(int)  
B_test  = B_test.astype(int)  
batch_size =len(X_train)  
print(A_train.shape, B_train.shape,B_test.shape )  
## rescale  
from sklearn.preprocessing import MinMaxScaler  
scaler = MinMaxScaler()  
# Train the Dataset  
X_train_scaled = scaler.fit_transform(A_train.astype(np.float65))  

"""