# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 21:10:36 2023

@author: GULO
"""

import cv2
import numpy as np
import dlib

image_path = "imgs/116.jpg"

im = cv2.imread(image_path)

predictor_path = "landmark/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
faces = detector(gray)
list_lms = []
for face in faces:
    # Predict facial landmarks
    landmarks = predictor(gray, face)
    # Draw the landmarks on the image
    for point in landmarks.parts():
        list_lms.append([point.y,point.x])
        
