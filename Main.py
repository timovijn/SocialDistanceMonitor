import cv2
import time
import numpy as np

# time.sleep(5)

vid_path = "./terrace1-c0.avi"
vid_cap = cv2.VideoCapture(vid_path)

(success, frame) = vid_cap.read()

(frame_h, frame_w) = (None, None)

count = 0
while count < 500:
    (success, frame) = vid_cap.read()
    (frame_h, frame_w) = frame.shape[:2]
    count += 1
    print(count)
    # cv2.imshow('frame', frame)
    # cv2.waitKey(1)

print('CP1')

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
labels = open(labelsPath).read().strip().split("\n")

frame_resized = cv2.resize(frame, (416, 416))
blob = cv2.dnn.blobFromImage(frame_resized, 1 / 255.0, (416, 416), swapRB=True, crop=False)         # Scale image by dividing by 255. YoloV3 needs input size (416, 416)
blobb = blob.reshape(blob.shape[2],blob.shape[3],3)
cv2.imshow('Blob', blobb)
cv2.waitKey(2000)

net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

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
                box = detection[0:4] * np.array([frame_w, frame_h, frame_w, frame_h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

print(idxs)
print(confidences)

print('CP2')

if len(idxs) > 0:
    status = []
    idf = idxs.flatten()
    close_pair = []
    s_close_pair = []
    centers = []
    co_info = []

    for i in idf:

        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        center = [int(x + w / 2), int(y + h / 2)]
        centers.append(center)
        cv2.circle(frame, tuple(center),1,(0,0,0),1)
        co_info.append([w, h, center])
        
        status.append(0)

print(centers)

print(x,y,h,w)

cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)

cv2.imshow('Social distancing analyser', frame)
cv2.waitKey(2000)

print('CP3')