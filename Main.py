# import cv2

vid_path = "./terrace1-c0.avi"
# vid = cv2.VideoCapture(vid_path)

# (vid_h, vid_w) = (None, None)
# (grabbed, frame) = vid.read()
# cv2.imshow('frame',1)

import cv2
import numpy as np
vidcap = cv2.VideoCapture(vid_path)

success,image = vidcap.read()

count = 0
while count < 500:
    success,image = vidcap.read()
    count += 1
    print(count)

    # cv2.imwrite("frame%d.jpg" % count, image)
success,image = vidcap.read()
cv2.imshow('frame', image)   
cv2.waitKey()
print('Done')
    # success,image = vidcap.read()
    # print('Read a new frame: ', success)
    # count += 1

# cap = cv2.VideoCapture("terrace1-c0.avi")
# ret, frame = cap.read()
# while(1):
#    ret, frame = cap.read()
#    cv2.imshow('frame',frame)
#    if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
#        cap.release()
#        cv2.destroyAllWindows()
#        break
#    cv2.imshow('frame',frame)