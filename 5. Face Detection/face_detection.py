# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 00:17:55 2023

@author: yusuf
"""

import cv2
import mediapipe as mp

#cap = cv2.VideoCapture('video2.mp4')
cap = cv2.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection() # tune inside the parantheses of FaceDetection() with values around 0.0-1.0
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(imgRGB)
    #print(results.detections)
    
    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            #print(bboxC)
    
            h, w, _ = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            
            cv2.rectangle(img, bbox, (0,255,255), thickness = 2)
    
    cv2.imshow('FaceDetection', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break