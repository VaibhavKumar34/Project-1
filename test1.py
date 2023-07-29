import cv2 as cv
import numpy as np


def empty(a):
    pass


def initTrackbars():
    cv.namedWindow('trackbars')
    cv.createTrackbar('nFrames', 'trackbars', 1, 30, empty)
    cv.createTrackbar('sF', 'trackbars', 11, 200, empty)
    cv.createTrackbar('mN', 'trackbars', 7, 100, empty)
    cv.createTrackbar('mS', 'trackbars', 7, 100, empty)


def readTrackbars():
    global sF, mN, mS, nFrames
    nFrames = cv.getTrackbarPos('nFrames', 'trackbars')
    sF = cv.getTrackbarPos('sF', 'trackbars')/10
    mN = cv.getTrackbarPos('mN', 'trackbars')
    mS = cv.getTrackbarPos('mS', 'trackbars')


initTrackbars()
url1 = 'videos/fps60/lab_amandeep.mp4'
url2 = 'videos/fps60/lab_barua.mp4'
url3 = 'videos/fps60/lab_roshan.mp4'
url4 = 'videos/fps60/lab_sundaram.mp4'
cap = cv.VideoCapture(url4)
# cap = cv.VideoCapture('videos/testVideo.mp4')
cascade1 = 'haarcascades/haarcascade_license_plate_rus_16stages.xml'
cascade2 = 'haarcascades/haarcascade_russian_plate_number.xml'
cascade3 = 'haarcascades/cars.xml'
haarcascade = cv.CascadeClassifier(cascade3)



while True:
    readTrackbars()
    _, img1 = cap.read()
    for i in range(nFrames):
        _, img2 = cap.read()
    if _ == False:
        break
    grayA = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    grayB = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    blurA = cv.GaussianBlur(grayA, (5, 5), 0)
    blurB = cv.GaussianBlur(grayB, (5, 5), 0)
    diffImg = cv.absdiff(blurA, blurB)
    facesRect = haarcascade.detectMultiScale(img1, scaleFactor=sF, minNeighbors=mN, minSize=(mS, mS))
    frame = img1.copy()
    for (x, y, w, h) in facesRect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.imshow('Frame', frame)
    cv.imshow('diffImg', diffImg)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()