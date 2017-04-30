import cv2
import numpy as np
import math

cap = cv2.VideoCapture("1.avi")

ret, frame = cap.read() # get a frame
w = int(cap.get(3))
h = int(cap.get(4))

all_frames = []
all_frames_RGB = []

threshold = 90

while(ret):
    new_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret, frame = cap.read()  
    if ret is False:
        break
    all_frames.append(new_frame)
    all_frames_RGB.append(frame)

#print (len(all_frames))
t=len(all_frames)
for i in range (t-1):
    m = cv2.absdiff(all_frames[i],all_frames[i+1]) #compare differences
    ret,B=cv2.threshold(m,threshold,255,cv2.THRESH_BINARY) #binary
    _,cnts,_ = cv2.findContours(B, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #detect edge
    
    for c in cnts:
        
        (x, y, w, h) = cv2.boundingRect(c)
        
        cv2.rectangle(all_frames_RGB[i], (x, y), (x+ w, y + h), (0,255,0), 1)
    cv2.imshow('1',all_frames_RGB[i])
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break
    
cap.release()
cv2.destroyAllWindows()
