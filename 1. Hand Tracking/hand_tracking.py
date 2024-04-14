# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 23:41:20 2023

@author: yusuf
"""

import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0) # 0 is the default camera

mpHand = mp.solutions.hands
hands = mpHand.Hands()
# static_image_mode boolean, tracking
# max_num_hands
# min_detection_confidence
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0


while True:
    success, img = cap.read()
    #imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flippedImg = cv2.flip(img,1)
    
    results = hands.process(flippedImg)
    print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(flippedImg, handLms, mpHand.HAND_CONNECTIONS)
            
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = flippedImg.shape
                
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # mark the wrist with a dot
                if id == 0:
                    cv2.circle(flippedImg, (cx, cy), 9, (255, 0, 0), cv2.FILLED)
    
    # fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(flippedImg, "FPS: " + str(int(fps)), (10, 30) , cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow('img', flippedImg)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
