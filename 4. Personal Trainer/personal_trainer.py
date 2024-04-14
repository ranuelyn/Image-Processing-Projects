# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 23:33:05 2023

@author: yusuf
"""

import cv2
import mediapipe as mp
import time
import numpy as np
import math

#cap = cv2.VideoCapture('video1.mp4')
cap = cv2.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

def findAngle(img, p1, p2, p3, lmList, draw = True):
    x1, y1 = lmList[p1][1:] # z x y points in the element on lmList
    x2, y2 = lmList[p2][1:]
    x3, y3 = lmList[p3][1:]
    
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    if angle < 0:
        angle += 360
        
    if draw:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.line(img, (x3, y3), (x2, y2), (0, 0, 255), 3)
        
        cv2.circle(img, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, (x3, y3), 10, (0, 255, 255), cv2.FILLED) # yellow circle on points

        cv2.circle(img, (x1, y1), 15, (0, 255, 255))
        cv2.circle(img, (x2, y2), 15, (0, 255, 255))
        cv2.circle(img, (x3, y3), 15, (0, 255, 255))        
        
        cv2.putText(img, str(int(angle)) + "'", (x2 - 53, y2 + 7), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    
    return angle

dir = 0
count = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    
    lmList = []
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
        
    #print(lmList)
    
    # check the angles
    if len(lmList) != 0:
        # push up
        p1, p2, p3 = 11, 13, 15
        angle = findAngle(img, p1, p2, p3, lmList)        
        per = np.interp(angle, (185, 245), (0, 100))
        #print(angle)    
        
        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1
                
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0
        
        print(count)
        
        cv2.putText(img, "Push Up Count: " + str(int(count)), (10, 30) , cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, "Push Up Count: " + str(int(count)), (10, 30) , cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow('Personal Trainer', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
