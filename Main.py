import cv2
import time
import numpy as np

# time.sleep(5)

vid_path = "./terrace1-c0.avi"
vidcap = cv2.VideoCapture(vid_path)

(success, frame) = vidcap.read()

(img_h, img_w) = (None, None)

count = 0
while count < 1000:
    (success, frame) = vidcap.read()
    (img_h, img_w) = frame.shape[:2]
    count += 1
    print(count)
    # cv2.imshow('frame', frame)
    # cv2.waitKey(1)

print('Done')

cv2.imshow('frame', frame)
cv2.waitKey(2000)

confid = 0.5
thresh = 0.5

wgt_path = "./yolov4.weights"
cfg_path = "./yolov4.cfg"
labelsPath = "./coco.names"

net = cv2.dnn.readNetFromDarknet(cfg_path, wgt_path)
ln = net.getLayerNames()
labels = open(labelsPath).read().strip().split("\n")

blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

boxes = []
confidences = []
classIDs = []

print(boxes)

for output in layerOutputs:
    print(output)
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if labels[classID] == "person":
            if confidence > confid:
                box = detection[0:4] * np.array([img_w, img_h, img_w, img_h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

print(box)
print('Done_2')
