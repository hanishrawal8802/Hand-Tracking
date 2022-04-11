import numpy as np
import cv2
import mediapipe as mp 
import time
import handtrmodule as htm
ptime = 0
ctime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True :
     
     success,img = cap.read()
     img = detector.findHands(img)
     Lmlist = detector.findPositions(img)
     if len(Lmlist) != 0:
       print(Lmlist[4])
     ctime = time.time()
     fps = 1/(ctime-ptime)
     ptime = ctime

     cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(240,0,240),3)
            

     cv2.imshow("Image",img)
     cv2.waitKey(1)