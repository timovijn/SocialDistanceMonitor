# Bruno Martens & Timo Vijn
# Social Distancing

#################### (Section) Preamble

########## (Subsection) Packages

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
from termcolor import colored
import matplotlib.pyplot as plt
from tkinter import Tk

########## (Subsection) Grab screen dimensions

screen_width = Tk().winfo_screenwidth()
screen_height = Tk().winfo_screenheight()
print(screen_width,screen_height)

########## (Subsection) Start timer

start_time = datetime.now()
print(''), print(colored('...','white')), print(''), print('Started at', start_time.strftime("%H:%M:%S"))


########## (Subsection) Choose video

# vid_path = "./video.mp4"
# vid_path = "./Videos/Pedestrian overpass - original video (sample) - BriefCam Syndex.mp4"
# vid_path = "./Videos/terrace1-c0.avi"
# vid_path = "./Videos/Delft.MOV"
vid_path = "./Videos/TownCentreXVID.avi"
# vid_path = "./Videos/WalkByShop1cor.mpg"
# vid_path = "./Videos/Rosmalen.MOV"

vid_cap = cv2.VideoCapture(vid_path)
vid_fps = vid_cap.get(cv2.CAP_PROP_FPS)

########## (Subsection) Choose clip region

clip_start_s = 11
clip_end_s = 15

clip_start = int(clip_start_s * vid_fps)
clip_end = int(clip_end_s * vid_fps)

########## (Subsection) Print video information

print(''), print(colored('...','white')), print(''), print(colored('Checkpoint', 'blue'),'Initialisation')
print(''), print(colored('...','white')), print(''), print('Path: {}'.format(vid_path)), print('Width: {} px'.format(int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))), print('Height: {} px'.format(int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))), print('Duration: {} s'.format(round(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)/vid_fps,2))), print('Framerate: {} fps'.format(vid_fps)), print('Frames: {}'.format(int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))))

########## (Subsection) Define clicking function

def get_mouse_points(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y
        cv2.circle(image, (x, y), 10, material.red(shade=50), 10)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        print("Point marked")
        print(x,y)

########## (Subsection) Initialise variables

mouse_pts = []
(frame_h, frame_w) = (None, None)
frame_num = 1
num_violations_cumulative = 0
num_pedestrians_cumulative = 0
frame_num = 0
heatmap_matrix = np.zeros((10,10))
total_pedestrians_detected = 0
total_six_feet_violations = 0
total_pairs = 0
abs_six_feet_violations = 0
pedestrian_per_sec = 0
sh_index = 1
sc_index = 1

########## (Subsection) Initialise windows

cv2.namedWindow("Perspective")
cv2.setMouseCallback("Perspective", get_mouse_points)
first_frame_display = True

scale_w = 1
scale_h = 1

SOLID_BACK_COLOR = material.gray(shade=90)

########## (Subsection) Setup video writer

height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_movie = cv2.VideoWriter("Pedestrian_detect.avi", fourcc, vid_fps, (width, height))
bird_movie = cv2.VideoWriter(
    "Pedestrian_bird.avi", fourcc, vid_fps, (int(width * scale_w), int(height * scale_h))
)

#################### (Section) Start

for frame_idx in range(clip_start, clip_end + 1):

    frame_num += 1

    if frame_idx > clip_end:
        break

    print(''), print(colored('...','white')), print(''), print(colored('New frame', 'green'),f'{frame_idx} ({clip_start} → {clip_end}) ({frame_num})')

    vid_cap.set(1, frame_idx)
    (success, frame) = vid_cap.read()
    (frame_h, frame_w) = frame.shape[:2]

    #################### (Section) Perspective

    if frame_idx == clip_start:
        # Ask user to mark parallel points and two points 6 feet apart. Order bl, br, tr, tl, p1, p2
        print(''), print(colored('...','white')), print(''), print(colored('Checkpoint', 'blue'),'Perspective')
        print(''),print(colored('...','white')),print(''),print('Mark (Bottom left) → (Bottom right) → (Top left) → (Top right)'),print(''),print(colored('...','white')),print('')
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
        bird_height = frame_h * scale_h
        bird_width = frame_w * scale_w
        bird_image = np.zeros(
            (int(bird_height), int(bird_width), 3), np.uint8
        )

        bird_image[:] = SOLID_BACK_COLOR
        pedestrian_detect = frame

        print(''),print(colored('...','white')),print(''),print(f'Threshold: {int(d_thresh)} px')

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
    # boxes = boxes[idxs.flatten()]
    boxes = np.array(boxes)[idxs.flatten()]
    boxes2 = np.array(boxes2)[idxs.flatten()]
    boxes_norm = np.array(boxes_norm)[idxs.flatten()]
    boxes_norm2 = np.array(boxes_norm2)[idxs.flatten()]
    confidences = np.array(confidences)[idxs.flatten()]
    classIDs = np.array(classIDs)[idxs.flatten()]

    print(''), print(colored('...','white')), print(''), print(colored('Checkpoint', 'blue'),'Object detection')
    print(''),print(colored('...','white')),print(''),print('Confidences:', [round(num, 2) for num in confidences])

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

    pts = np.array(
        [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32
    )
    print(pts)
    cv2.polylines(frame, [pts], True, (0, 255, 255), thickness=4)

    pedestrian_boxes = boxes_norm2
    num_pedestrians = len(boxes_norm2)
    # Detect person and bounding boxes using DNN
    # pedestrian_boxes, num_pedestrians = DNN.detect_pedestrians(frame)

    if len(pedestrian_boxes) > 0:
        print(''), print(colored('...','white')), print(''), print(colored('Checkpoint', 'blue'),'Social distancing'), print(''), print(colored('...','white')), print('')
        # pedestrian_detect = plot_pedestrian_boxes_on_image(frame, pedestrian_boxes)
        warped_pts, bird_image, pairs, num_violations, dd, heatmap_matrix, frame = plot_points_on_bird_eye_view(
            frame, pedestrian_boxes, M, scale_w, scale_h,d_thresh, heatmap_matrix, bird_height, bird_width
        )
        # six_feet_violations, ten_feet_violations, pairs = plot_lines_between_nodes(
        #     warped_pts, bird_image, d_thresh
        # )
        # plot_violation_rectangles(pedestrian_boxes, )

        ###########################

        ###########################

        num_violations_cumulative += num_violations
        num_pedestrians_cumulative += num_pedestrians

        # total_six_feet_violations += violations / vid_fps
        # abs_six_feet_violations += violations
        # pedestrian_per_sec, sh_index = calculate_stay_at_home_index(
        #     total_pedestrians_detected, frame_num, vid_fps
        # )
        print(f'Pedestrians: {num_pedestrians} ({num_pedestrians_cumulative})')
        print(f'Pairs: {pairs}')
        print(f'Violating pairs: {dd}')
        print(f'Violations: {num_violations} ({num_violations_cumulative})')

        print(''), print(colored('...','white')), print(''), print(colored('Checkpoint', 'blue'),'Social distancing performance'), print(''), print(colored('...','white')), print('')
        print(f'Frames: {frame_num}')
        print(f'Violations: {round(num_violations_cumulative/frame_num,1)}')
        print(f'Pedestrians: {round(num_pedestrians_cumulative/frame_num,1)}')

    # last_h = 75
    # text = "# 6ft violations: " + str(int(total_six_feet_violations))
    # pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

    # text = "Stay-at-home Index: " + str(np.round(100 * sh_index, 1)) + "%"
    # pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

    # if total_pairs != 0:
    #     sc_index = 1 - abs_six_feet_violations / total_pairs

    # text = "Social-distancing Index: " + str(np.round(100 * sc_index, 1)) + "%"
    # pedestrian_detect, last_h = put_text(pedestrian_detect, text, text_offset_y=last_h)

    # uniform_data = np.random.rand(10, 10)
    # print(uniform_data)
    # seaborn.heatmap(uniform_data)
    # plt.show()

    # print(heatmap_matrix)

    # cv2.imshow("Perspective", pedestrian_detect)
    # output_movie.write(pedestrian_detect)
    # bird_movie.write(bird_image)
    # cv2.waitKey(0)

    # plt.figure(figsize=[width,height])

end_time = datetime.now()

seaborn.heatmap(heatmap_matrix)
plt.show()