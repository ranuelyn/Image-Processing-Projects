# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:10:59 2023

@author: yusuf
"""

import cv2
import mediapipe as mp
import time 

mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('video5.mp4')
#cap = cv2.VideoCapture(0)

cTime = 0
pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            
            # make a circle to the elbows
            if id == 13:
                cv2.circle(img, (cx, cy), 5, (0, 0, 0), cv2.FILLED)
            if id == 14:
                cv2.circle(img, (cx, cy), 5, (0, 0, 0), cv2.FILLED)
        
    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 30) , cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow('Pose Estimation', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
