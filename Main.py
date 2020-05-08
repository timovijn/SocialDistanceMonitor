# Bruno Martens & Timo Vijn

import os
os.system('clear')

import cv2
import time
import numpy as np
import imutils

from datetime import datetime
start_time = datetime.now()

print(''), print('...'), print(''), print('Started at', start_time.strftime("%H:%M:%S"))

# vid_path = "./video.mp4"
# vid_path = "./Videos/terrace1-c0.avi"
# vid_path = "./Videos/Delft.MOV"
# vid_path = "./Videos/TownCentreXVID.avi"
# vid_path = "./Videos/WalkByShop1cor.mpg"

##########################
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# ffmpeg_extract_subclip(vid_path, 10, 20, targetname="clip.mp4")
# clip_path = "clip.mp4"
##########################

(frame_h, frame_w) = (None, None)

vid_cap = cv2.VideoCapture(vid_path)
vid_fps = vid_cap.get(cv2.CAP_PROP_FPS)

print(''), print('...'), print(''), print('Path: {}'.format(vid_path)), print('Width: {} px'.format(int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))), print('Height: {} px'.format(int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))), print('Duration: {} s'.format(round(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)/vid_fps,2))), print('Framerate: {} fps'.format(vid_fps)), print('Frames: {}'.format(int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))))

clip_start = int(20 * vid_fps)
clip_end = int(21 * vid_fps)

for frame_count in range(clip_start, clip_end + 1):

    if frame_count > clip_end:
        break

    print('')
    print('...')
    print('')

    print('Current frame:', frame_count)

    vid_cap.set(1, frame_count)
    (success, frame) = vid_cap.read()
    frame = imutils.resize(frame, width = 1920)
    (frame_h, frame_w) = frame.shape[:2]

    confid = 0.5
    thresh = 0.5

    wgt_path = "./Yolo/yolov3.weights"
    cfg_path = "./Yolo/yolov3.cfg"
    labelsPath = "./Yolo/coco.names"

    net = cv2.dnn.readNetFromDarknet(cfg_path, wgt_path)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    labels = open(labelsPath).read().strip().split("\n")

    frame_resized = cv2.resize(frame, (416, 416))                                                   # Scale image by dividing by 255. YoloV3 needs input size (416, 416)
    blob = cv2.dnn.blobFromImage(
        frame_resized, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    blobb = blob.reshape(blob.shape[2], blob.shape[3], 3)
    # cv2.imshow('Blob', blobb)
    # cv2.waitKey(2)

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
                    box = detection[0:4] * \
                        np.array([frame_w, frame_h, frame_w, frame_h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

    print('Confidences:', confidences)
    print('(CP2)')

    if len(idxs) > 0:
        status = []
        idf = idxs.flatten()
        close_pair = []
        s_close_pair = []
        centers = []
        co_info = []
        X = []
        Y = []
        W = []
        H = []

        for i in idf:

            (x, y) = (boxes[i][0], boxes[i][1])
            X.append(x)
            Y.append(y)
            (w, h) = (boxes[i][2], boxes[i][3])
            W.append(w)
            H.append(h)
            center = [int(x + w / 2), int(y + h / 2)]
            centers.append(center)
            cv2.circle(frame, tuple(center), 1, (0, 0, 0), 1)
            co_info.append([w, h, center])

            status.append(0)

        print('Centers:', centers)

        for i in range(0, len(idf)):
            cv2.rectangle(
                frame, (X[i], Y[i]), (X[i] + W[i], Y[i] + H[i]), (0, 0, 150), 2)
        cv2.imshow('Person recognition', frame)
        cv2.waitKey(1)

    print('(CP3)')

end_time = datetime.now()

print('')
print('...')
print('')
print('Finished at {}'.format(end_time.strftime("%H:%M:%S")), '({})'.format(datetime.strptime(end_time.strftime("%H:%M:%S"), "%H:%M:%S") - datetime.strptime(start_time.strftime("%H:%M:%S"), "%H:%M:%S")))
print('')
print('...')
print('')