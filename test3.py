import cv2 as cv
import numpy as np

url1 = 'videos/fps60/lab_amandeep.mp4'
url2 = 'videos/fps60/lab_barua.mp4'
url3 = 'videos/fps60/lab_roshan.mp4'
url4 = 'videos/fps60/lab_sundaram.mp4'
cap = cv.VideoCapture(url2)

# adjusting frame rate
fps = cap.set(cv.CAP_PROP_FPS, 1)

# minimum contour width
min_contour_width = 40  # 40

# minimum contour height
min_contour_height = 40  # 40
offset = 10  # 10
line_height = 350  # 550
matches = []
cars = 0

cascade1 = 'haarcascades/haarcascade_license_plate_rus_16stages.xml'
cascade2 = 'haarcascades/haarcascade_russian_plate_number.xml'
cascade3 = 'haarcascades/cars.xml'
haarcascade = cv.CascadeClassifier(cascade3)


def empty(a):
    pass


def initTrackbars():
    cv.namedWindow('trackbars')
    cv.createTrackbar('nFrames', 'trackbars', 1, 30, empty)
    cv.createTrackbar('sF', 'trackbars', 11, 200, empty)
    cv.createTrackbar('mN', 'trackbars', 4, 100, empty)
    cv.createTrackbar('mS', 'trackbars', 4, 100, empty)


def readTrackbars():
    global sF, mN, mS, nFrames
    nFrames = cv.getTrackbarPos('nFrames', 'trackbars')
    sF = cv.getTrackbarPos('sF', 'trackbars')/10
    mN = cv.getTrackbarPos('mN', 'trackbars')
    mS = cv.getTrackbarPos('mS', 'trackbars')


def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1
    return cx, cy
    # return [cx, cy]


# cap.set(3, 1280)
# cap.set(4, 720)
initTrackbars()

while True:
    readTrackbars()
    _, frame1 = cap.read()
    _, frame2 = cap.read()
    if not _:
        print('Error in reading OR End of video!!!')
        break
    d = cv.absdiff(frame1, frame2)
    grey = cv.cvtColor(d, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(grey, (5, 5), 0)

    ret, th = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(th, np.ones((3, 3)))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))

    # Fill any small holes
    closing = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)
    contours, h = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    detected = haarcascade.detectMultiScale(closing, scaleFactor=sF, minNeighbors=mN, minSize=(mS, mS))
    for (i, c) in enumerate(contours):
    # for (x, y, w, h) in detected:
        (x, y, w, h) = cv.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)

        if not contour_valid:
            continue
        cv.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)

        cv.line(frame1, (0, line_height), (1280, line_height), (0, 255, 0), 2)
        centroid = get_centroid(x, y, w, h)
        matches.append(centroid)
        cv.circle(frame1, centroid, 5, (0, 255, 0), -1)
        cx, cy = get_centroid(x, y, w, h)
        for (x, y) in matches:
            if (line_height + offset) > y > (line_height - offset):
                cars = cars + 1
                matches.remove((x, y))
                print(cars)

    cv.putText(frame1, "Total Vehicles Detected: " + str(cars), (10, 90), cv.FONT_HERSHEY_SIMPLEX, 1,
                (0, 170, 0), 2)

    # cv.drawContours(frame1, contours, -1, (0, 0, 255), 2)

    cv.imshow("OUTPUT", frame1)
    # cv.imshow("Difference", th)
    if cv.waitKey(1) == 27:
        break

# print(matches)
cv.destroyAllWindows()
cap.release()
