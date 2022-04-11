import numpy as np
import cv2
import mediapipe as mp 
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity = 1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon


        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,
                                        self.detectionCon, self.trackCon,)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw = True):

     imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
     self.results = self.hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
     if self.results.multi_hand_landmarks:
        for handlms in self.results.multi_hand_landmarks:
          if draw:
            self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)
     return img

               
    def findPositions(self,img,handNO=0,draw = True):
        Lmlist =[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNO]
            for id,lm in enumerate(myHand.landmark):
                #print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                #print(id,cx,cy)
                Lmlist.append([id,cx,cy])
                #if id == 4:
                if draw:
                 cv2.circle(img,(cx,cy),8,(255,0,0),cv2.FILLED)
        return Lmlist

  

def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

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



if __name__ == "__main__":
    main()
        