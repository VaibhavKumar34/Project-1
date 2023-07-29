import cv2 as cv
import numpy as np

def empty(a):
    pass


def initTrackbars():
    cv.namedWindow('trackbars')
    cv.createTrackbar('threshold1', 'trackbars', 100, 1000, empty)
    cv.createTrackbar('threshold2', 'trackbars', 350, 1000, empty)


def readTracbars():
    global thrld1, thrld2
    thrld1 = cv.getTrackbarPos('threshold1', 'trackbars')
    thrld2 = cv.getTrackbarPos('threshold2', 'trackbars')


path = 'testImage0.jpeg'
img = cv.imread(path)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
initTrackbars()


while True:
    readTracbars()
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    edges = cv.Canny(blur, threshold1=thrld1, threshold2=thrld2)
    cntrs, heir = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, cntrs, -1, (255, 0, 0), 2)
    cv.imshow('edges', edges)
    cv.imshow('img', img)
    cv.waitKey(1)
    