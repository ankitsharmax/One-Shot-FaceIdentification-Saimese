# Removing warnings
import warnings
warnings.filterwarnings('ignore')

import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import imutils
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from utils import LRN2D
import utils
import model
import process

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

model = model.model_structure()
model.load_weights('check')

def face_capture(userName):
    cam = cv2.VideoCapture(0)

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    count = 0
    while(True):
        ret, img = cam.read()
        cv2.imshow("image",img)
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), 2)     
            count += 1
            # Save the captured image into the datasets folder
            if count == 10:
                cv2.imwrite(f"images/{userName}" + ".jpg", img[y1:y2,x1:x2])
        k = cv2.waitKey(200) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30: # Take face sample and stop video
             break
    cam.release()
    cv2.destroyAllWindows()

def face_capture_image(userName,imgPath):
    imgPath = f"addUser/{imgPath}"
    img = cv2.imread(imgPath)
    WIDTH = 600
    img = imutils.resize(img,width=WIDTH)

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = face_detector.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in faces:
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),2)

        cv2.imwrite(f"images/{userName}"+".jpg",img[y1:y2,x1:x2])
        print(f"User Added {userName}")
        cv2.destroyAllWindows()


user = input("Enter name of the user ")
z = os.listdir("images/")
if f"{user}.jpg" not in z:
    face_capture(user)
else:
    print("User already present in the database")
    option = int(input("\nChoose option\n1.Update\n2.Cancel\n\n"))
    if option == 1:
        face_capture(user)

# path = f'images/{user}.jpg'
# im = plt.imread(path)
# plt.xticks([])
# plt.yticks([])
# plt.title(user)
# plt.imshow(im)


user = input("Enter name of the user ")
z = os.listdir("images/")
if f"{user}.jpg" not in z:
    print("\n#### Put your image in the addUser directory####\n")
    imageName = input("Enter the name of the image with type\nExample\n- image.jpg\n\nImage Name: ")
    face_capture_image(userName=user,imgPath=imageName)
else:
    print("User already present in the database")
    option = int(input("\nChoose option\n1.Update\n2.Cancel\n\n"))
    if option == 1:
        face_capture_image(userName=user,imgPath=imageName)

input_embeddings = process.create_input_image_embeddings(model)
process.recognize_faces_in_cam(input_embeddings,model)

