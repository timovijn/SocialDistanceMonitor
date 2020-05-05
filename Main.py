import cv2
import time
import numpy as np

# time.sleep(5)

vid_path = "./terrace1-c0.avi"
vidcap = cv2.VideoCapture(vid_path)

success,image = vidcap.read()

count = 0
while count < 1000:
    success,image = vidcap.read()
    count += 1
    print(count)
    cv2.imshow('frame', image)
    cv2.waitKey(1)

print('Done')