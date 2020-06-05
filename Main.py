# Bruno Martens & Timo Vijn

import os
os.system('clear')

import cv2
import time
import numpy as np
import imutils
from myfunctions import *
from datetime import datetime
from SecretColors.palette import Palette
material = Palette("material",color_mode="rgb255")

start_time = datetime.now()
print(''), print('...'), print(''), print('Started at', start_time.strftime("%H:%M:%S"))

# vid_path = "./video.mp4"
# vid_path = "./Videos/Pedestrian overpass - original video (sample) - BriefCam Syndex.mp4"
# vid_path = "./Videos/terrace1-c0.avi"
# vid_path = "./Videos/Delft.MOV"
vid_path = "./Videos/TownCentreXVID.avi"
# vid_path = "./Videos/WalkByShop1cor.mpg"
# vid_path = "./Videos/Rosmalen.MOV"

clip_start_s = 100
clip_end_s = 200

##########################
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# ffmpeg_extract_subclip(vid_path, 10, 20, targetname="clip.mp4")
# clip_path = "clip.mp4"
##########################

##########################
##########################
mouse_pts = []

def get_mouse_points(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y
        cv2.circle(image, (x, y), 10, material.red(shade=50), 10)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        print("Point detected")
        print(mouse_pts)
##########################
##########################

(frame_h, frame_w) = (None, None)

vid_cap = cv2.VideoCapture(vid_path)
vid_fps = vid_cap.get(cv2.CAP_PROP_FPS)

print(''), print('...'), print(''), print('Path: {}'.format(vid_path)), print('Width: {} px'.format(int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))), print('Height: {} px'.format(int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))), print('Duration: {} s'.format(round(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)/vid_fps,2))), print('Framerate: {} fps'.format(vid_fps)), print('Frames: {}'.format(int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))))

clip_start = int(clip_start_s * vid_fps)
clip_end = int(clip_end_s * vid_fps)

##########################
##########################
frame_num = 1
total_pedestrians_detected = 0
total_six_feet_violations = 0
total_pairs = 0
abs_six_feet_violations = 0
pedestrian_per_sec = 0
sh_index = 1
sc_index = 1

cv2.namedWindow("Perspective")
cv2.setMouseCallback("Perspective", get_mouse_points)
num_mouse_points = 0
first_frame_display = True

scale_w = 1 # 1.2 / 2
scale_h = 2 # 4 / 2

SOLID_BACK_COLOR = material.gray(shade=90)

# Setup video writer
cap = cv2.VideoCapture(vid_path)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_movie = cv2.VideoWriter("Pedestrian_detect.avi", fourcc, fps, (width, height))
bird_movie = cv2.VideoWriter(
    "Pedestrian_bird.avi", fourcc, fps, (int(width * scale_w), int(height * scale_h))
)

##########################
##########################

for frame_count in range(clip_start, clip_end + 1):

    if frame_count > clip_end:
        break

    print('')
    print('...')
    print('')

    print(f'Current frame: {frame_count} ({clip_start} → {clip_end})')

    vid_cap.set(1, frame_count)
    (success, frame) = vid_cap.read()
    frame = imutils.resize(frame, width = 1920)
    (frame_h, frame_w) = frame.shape[:2]

    if frame_count == clip_start:
        # Ask user to mark parallel points and two points 6 feet apart. Order bl, br, tr, tl, p1, p2
        print(''),print('...'),print(''),print('Mark (Bottom left) → (Bottom right) → (Top left) → (Top right)'),print(''),print('...'),print('')
        while True:
            image = frame
            cv2.imshow("Perspective", image)
            cv2.waitKey(1)
            if len(mouse_pts) == 7:
                cv2.destroyWindow("Perspective")
                break
            first_frame_display = False
        four_points = mouse_pts

        # Get perspective
        M, Minv = get_camera_perspective(frame, four_points[0:4])
        pts = src = np.float32(np.array([four_points[4:]]))
        warped_pt = cv2.perspectiveTransform(pts, M)[0]
        d_thresh = np.sqrt(
            (warped_pt[0][0] - warped_pt[1][0]) ** 2
            + (warped_pt[0][1] - warped_pt[1][1]) ** 2
        )
        bird_image = np.zeros(
            (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
        )

        bird_image[:] = SOLID_BACK_COLOR
        pedestrian_detect = frame

        print(''),print('...'),print(''),print(f'Threshold: {int(d_thresh)} px'),print(''),print('...'),print('')

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
    boxes2 = []
    boxes_norm = []
    boxes_norm2 = []
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
                    x_l = centerX - (width / 2)
                    x_r = centerX + (width / 2)
                    y_t = centerY - (height / 2)
                    y_b = centerY + (height / 2)
                    boxes.append([x, y, int(width), int(height)])
                    boxes2.append([y_t,x_l,y_b,x_r])
                    boxes_norm.append([x/frame_w, y/frame_h, int(width)/frame_w, int(height)/frame_h])
                    boxes_norm2.append([y_t/frame_h,x_l/frame_w,y_b/frame_h,x_r/frame_w])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
    print(f'Indexes: {idxs}')
    # boxes = boxes[idxs.flatten()]
    print(idxs.flatten())
    
    print(f'Boxes (old): {boxes}')
    boxes = np.array(boxes)[idxs.flatten()]
    boxes2 = np.array(boxes2)[idxs.flatten()]
    boxes_norm = np.array(boxes_norm)[idxs.flatten()]
    boxes_norm2 = np.array(boxes_norm2)[idxs.flatten()]
    confidences = np.array(confidences)[idxs.flatten()]
    classIDs = np.array(classIDs)[idxs.flatten()]
    print(f'Boxes (new): {boxes}')

    print('Confidences:', [round(num, 2) for num in confidences])
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

        for i in range(0, len(idf)):

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

        for i in range(0, len(idf)):
            cv2.rectangle(
                frame, (X[i], Y[i]), (X[i] + W[i], Y[i] + H[i]), (0, 0, 150), 2)
        cv2.imshow('Person recognition', frame)
        cv2.waitKey(1)
    
    print('Centers:', centers)
    print('(CP3)')

    pts = np.array(
        [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32
    )
    cv2.polylines(frame, [pts], True, (0, 255, 255), thickness=4)

    pedestrian_boxes = boxes_norm2
    num_pedestrians = len(boxes_norm2)
    print(f'Boxes norm: {len(boxes_norm2)}')
    print(f'Boxes: {len(boxes)}')
    print(f'Indexes: {len(idxs)}')
    # Detect person and bounding boxes using DNN
    # pedestrian_boxes, num_pedestrians = DNN.detect_pedestrians(frame)

    if len(pedestrian_boxes) > 0:
        # pedestrian_detect = plot_pedestrian_boxes_on_image(frame, pedestrian_boxes)
        warped_pts, bird_image = plot_points_on_bird_eye_view(
            frame, pedestrian_boxes, M, scale_w, scale_h,d_thresh
        )
        six_feet_violations, ten_feet_violations, pairs = plot_lines_between_nodes(
            warped_pts, bird_image, d_thresh
        )
        # plot_violation_rectangles(pedestrian_boxes, )
        total_pedestrians_detected += num_pedestrians
        total_pairs += pairs

        total_six_feet_violations += six_feet_violations / vid_fps
        abs_six_feet_violations += six_feet_violations
        pedestrian_per_sec, sh_index = calculate_stay_at_home_index(
            total_pedestrians_detected, frame_num, vid_fps
        )

    last_h = 75
    text = "# 6ft violations: " + str(int(total_six_feet_violations))
    pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

    text = "Stay-at-home Index: " + str(np.round(100 * sh_index, 1)) + "%"
    pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

    if total_pairs != 0:
        sc_index = 1 - abs_six_feet_violations / total_pairs

    text = "Social-distancing Index: " + str(np.round(100 * sc_index, 1)) + "%"
    pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

    cv2.imshow("Perspective", pedestrian_detect)
    output_movie.write(pedestrian_detect)
    bird_movie.write(bird_image)
    # cv2.waitKey(0)

end_time = datetime.now()

print('')
print('...')
print('')
print('Finished at {}'.format(end_time.strftime("%H:%M:%S")), '({})'.format(datetime.strptime(end_time.strftime("%H:%M:%S"), "%H:%M:%S") - datetime.strptime(start_time.strftime("%H:%M:%S"), "%H:%M:%S")))
print('')
print('...')
print('')
