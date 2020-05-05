import cv2
import numpy as np

cap = cv2.VideoCapture('Data/terrace1-c0.avi')

if cap.isOpened() == False:
    print("Error opening video file")

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Frame', frame)
    elif cv2.waitKey(1) == ord('q'):
        break
    else:
        break

cap.release()
cv2.destroyAllWindows()
