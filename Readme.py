import cv2
import numpy as np
import math
import argparse

#analyze parameter
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

cap = cv2.VideoCapture("1.avi")
firstFrame= None

threshold = 10

while True:
    # get a frame
    (grabbed, frame) = cap.read()
    if not grabbed:
        break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if firstFrame is None:
        firstFrame = gray
        continue
    #calculate diff
    frameDelta = cv2.absdiff(firstFrame,gray)
    thresh = cv2.threshold(frameDelta,threshold,255,cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh,None,iterations=2)
    _,cnts,_=cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #analyze all contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
        #compute the bounding box and draw
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF
 
    # press 'q', quit
    if key == ord("q"):
        break
 
cap.release()
cv2.destroyAllWindows()


