# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 01:25:22 2023

@author: yusuf
"""

import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture('video2.mp4')

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 1)

mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)

pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    
    print(results.multi_face_landmarks) # check the landmarks
    
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, facelms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec) # FACEMESH_CONTOURS
            
        for id, lm in enumerate(facelms.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            print([id, cx, cy])
    
    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 30) , cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 30) , cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    
    
    cv2.imshow('FaceMesh', img)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break