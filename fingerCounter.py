import cv2
import numpy as np
import  time
import handTrackingModule as htm
import os



cap =  cv2.VideoCapture(0)
widthCam ,heightCam = 640 , 480
cap.set(3 , widthCam)
cap.set(4 , heightCam)
previous_time = 0

folderPath = "images/hands"
imageList = os.listdir(folderPath)
overlayList = []
for imagePath in imageList:
    image = cv2.imread(f'{folderPath}/{imagePath}')
    resized = cv2.resize(image, (200,200), interpolation = cv2.INTER_AREA)
    overlayList.append(resized)

detector = htm.HandDetector(detConfidence= 0.7)
tipsIds = [ 4 , 8 , 12 ,16 ,20]
while True:
    success ,img = cap.read()
    L_R_hand,img = detector.findHands(img)
    left_fingers_count ,right_fingers_count = 0 , 0
    leftFingers = []
    rightFingers = []
    if L_R_hand== "Left":
        landmarkList = detector.findPosition(img,draw=False)
        if len(landmarkList) !=0:
            rightFingers = detector.fingerUp()
            right_fingers_count = rightFingers.count(1)
            img[0:200 , 0:200] = overlayList[right_fingers_count]
    if L_R_hand== "Right":
        landmarkList = detector.findPosition(img, draw=False)
        if len(landmarkList) !=0:
            leftFingers = detector.fingerUp()
            left_fingers_count = leftFingers.count(1)
            img[0:200 , 200:400] = overlayList[left_fingers_count]
    if left_fingers_count != 0 or right_fingers_count !=0:
        fingers_count = left_fingers_count + right_fingers_count
        cv2.rectangle(img,(20,225) , (170, 425) , (255 , 0 ,0 ) , 4)
        cv2.putText(img,str(fingers_count) , (43,375) , cv2.FONT_HERSHEY_PLAIN , 10 , (255 , 0 ,0) ,10)
    current_time = time.time() 
    fbs = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, f'fbs: {int (fbs)}',(400,30), cv2.FONT_HERSHEY_PLAIN , 2,(255,0,0) , 2)
    cv2.imshow("Image" , img)
    cv2.waitKey(1)