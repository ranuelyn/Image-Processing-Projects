# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 00:27:04 2023

@author: yusuf
"""

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    fImg = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(fImg, cv2.COLOR_BGR2RGB)
    
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    
    
    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(fImg, handLms, mpHand.HAND_CONNECTIONS)
            
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = fImg.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                
                # index finger tip -> 8
                #if id == 8:
                #    cv2.circle(fImg, (cx, cy), 9, (255, 0, 0), cv2.FILLED)
                # index finger mid -> 6
                #if id == 6:
                #    cv2.circle(fImg, (cx, cy), 9, (0, 0, 255), cv2.FILLED)
                
    if len(lmList) != 0:
        
        fingers = []
        
        # thumb finger
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        
        else:
            fingers.append(0)
        
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        #print(fingers)        
        
        totalFingers = fingers.count(1)
        
        cv2.putText(fImg, "Finger Count: " + str(int(totalFingers)), (10, 30) , cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)

     
    cv2.imshow("Finger Counter", fImg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
