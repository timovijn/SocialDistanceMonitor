import cv2
import time
import numpy as np

# time.sleep(5)

vid_path = "./terrace1-c0.avi"
# vid_path = "./video.mp4"

# vid = vid_path

##########################
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# ffmpeg_extract_subclip(vid_path, 10, 20, targetname="clip.mp4")
# clip_path = "clip.mp4"
##########################

clip_start = 800
clip_end = 1000

frame_count = 0

(frame_h, frame_w) = (None, None)

vid_cap = cv2.VideoCapture(vid_path)

(success, frame) = vid_cap.read()

while(True):

    print('frame_count=',frame_count)

    if frame_count >= clip_start and frame_count <= clip_end:

        (success, frame) = vid_cap.read()
        (frame_h, frame_w) = frame.shape[:2]

        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)

        print('CP1')

        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

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
        cv2.waitKey(1)

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
            print('idf =', idf)
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
                cv2.circle(frame, tuple(center),1,(0,0,0),1)
                co_info.append([w, h, center])
                
                status.append(0)

            print(centers)
            print(X)
            print(Y)

            ilist = []
            i = 0

            for i in range(0,len(idf)):
                ilist.append(i)
                print(i)
                cv2.rectangle(frame, (X[i], Y[i]), (X[i] + W[i], Y[i] + H[i]), (0, 0, 150), 2)
            cv2.imshow('Social distancing analyser', frame)
            cv2.waitKey(1)


        print('CP3')

    else:
        (success, frame) = vid_cap.read()

    frame_count += 1