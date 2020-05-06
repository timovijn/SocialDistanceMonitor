import cv2
import time
import numpy as np

# time.sleep(5)

vid_path = "./video.mp4"
vidcap = cv2.VideoCapture(vid_path)

(success, frame) = vidcap.read()

(img_h, img_w) = (None, None)

count = 0
while count < 500:
    (success, frame) = vidcap.read()
    (img_h, img_w) = frame.shape[:2]
    count += 1
    print(count)
    # cv2.imshow('frame', frame)
    # cv2.waitKey(1)

print('Done')

cv2.imshow('Frame', frame)
cv2.waitKey(2000)

confid = 0.5
thresh = 0.5

wgt_path = "./yolov3.weights"
cfg_path = "./yolov3.cfg"
labelsPath = "./coco.names"

net = cv2.dnn.readNetFromDarknet(cfg_path, wgt_path)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(ln)
labels = open(labelsPath).read().strip().split("\n")

frame_resized = cv2.resize(frame, (416, 416))
blob = cv2.dnn.blobFromImage(frame_resized, 1 / 255.0, (416, 416), swapRB=True, crop=False)         # Scale image by dividing by 255. YoloV3 needs input size (416, 416)
print(blob)
print(blob.shape)
blobb = blob.reshape(blob.shape[2],blob.shape[3],3)
print(blobb.shape)
cv2.imshow('Blob', blobb)
cv2.waitKey(20000)

net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

print(layerOutputs)

boxes = []
confidences = []
classIDs = []

for output in layerOutputs:
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

print('Done_2')