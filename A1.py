from ultralytics import YOLO
import cv2 as cv
import cvzone as cvn
from sort import *
import numpy as np

cap = cv.VideoCapture('videos/testVideo.mp4')   # loading the video stream
mask = cv.imread('videos/mask.png')     # loading the mask for detection in the particular region
track = Sort(max_age=15, min_hits=3, iou_threshold=0.3)     # a program for tracking the detected vechiles
limits = [0, 300, 800, 300]
cntBus = []
cntCar = []
cntTruck = []
cntBike = []

model = YOLO('Yolo/yolov8n.pt')     # loading the ML model responsible for detection
classNames = model.names
# print(classNames)

while True:
    _, img = cap.read()
    # making the video stream compatible for detection 
    imgMask = cv.bitwise_and(img, mask)
    result = model(imgMask, stream=True)
    detected = np.empty((0, 5))
    # print(type(result))
    for r in result:
        box = r.boxes
        for b in box:
            x1, y1, x2, y2 = b.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            prob = round(float(b.conf[0]), 2)
            ctgry = int(b.cls[0])
            currCat = classNames[ctgry]
            # detecting the required types of vechiles in the video stream
            if currCat == 'car' or currCat == 'truck' or currCat == 'bus' or currCat == 'motorbike' and prob >0.1:
                cvn.cornerRect(img, (x1, y1, w, h), l=10, rt=5)
                cvn.putTextRect(img, f'{classNames[ctgry]} {prob}', (max(0, x1), max(40, y1)),
                                offset=3, scale=0.7, thickness=1)
                currArr = np.array((x1, y1, x2, y2, prob)) 
                detected = np.vstack((detected, currArr))
    resultTrack = track.update(detected)
    cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for r in resultTrack:
        x1, y1, x2, y2, id = r 
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        cvn.cornerRect(img, (x1, y1, w, h), l=10, rt=2, colorR=(0, 255, 0))
        # cvn.putTextRect(img, f'{id}', (max(0, x1), max(40, y1)),
        #                         offset=10, scale=2 , thickness=3)
        cx, cy = x1+w//2, y1+h//2
        cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)
        # counting the numbers of each type of vechiles. Classified vehicular flow
        if limits[0]<cx<limits[2] and limits[1]-15<cy<limits[3]+15:
            if cntBus.count(id) == 0:
                cntBus.append(id)
                cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 255), 5)
            if cntCar.count(id) == 0:
                cntCar.append(id)
                cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 255), 5)
            if cntTruck.count(id) == 0:
                cntTruck.append(id)
                cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 255), 5)
            if cntBike.count(id) == 0:
                cntBike.append(id)
                cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 255), 5)
        cvn.putTextRect(img, f'Count: {len(cntBus)}', (50, 50), scale=1, thickness=1)
        cvn.putTextRect(img, f'Count: {len(cntTruck)}', (200, 50), scale=1, thickness=1)
        cvn.putTextRect(img, f'Count: {len(cntCar)}', (400, 50), scale=1, thickness=1)
        cvn.putTextRect(img, f'Count: {len(cntBike)}', (600, 50), scale=1, thickness=1)
    # showing the detection 
    cv.imshow("Image", imgMask)
    cv.imshow("Img", img)
    cv.waitKey(1)



