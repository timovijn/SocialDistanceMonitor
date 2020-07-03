########################################
########################################
########################################
########################################
# Bruno Martens & Timo Vijn
# Social Distance Monitor
########################################
########################################
########################################
########################################

#################### (Section) Packages

import os
os.system('clear')
import cv2
import time
import numpy as np
import imutils
from datetime import datetime
from SecretColors.palette import Palette
material = Palette("material",color_mode="rgb255")
from termcolor import colored
import matplotlib.pyplot as plt
from tkinter import Tk
from scipy.spatial.distance import pdist, squareform
import seaborn
from PIL import Image
import PySimpleGUI as sg
from simple_term_menu import TerminalMenu
import time

#################### (Section) User settings

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

########## (Subsection) Choose video

vid_paths = []

vid_paths = [
    "./Videos/TownCentreXVID.avi",
    "./Videos/virat.mp4",
    "./Videos/Pedestrian overpass - original video (sample) - BriefCam Syndex.mp4",
    "./Videos/terrace1-c0.avi",
    "./Videos/Delft.MOV",
    "./Videos/WalkByShop1cor.mpg",
    "./Videos/Rosmalen.MOV",
    "./Videos/TownCentreXVID_240.mp4",
    "./Videos/TownCentreXVID_480.m4v",
    "./Videos/TownCentreXVID_960.m4v",
    "./Videos/TownCentreXVID_1920.m4v",
    "./Videos/Training.mov",
    "./Videos/INRIA_Train.avi",
    "./Videos/INRIA_Test.avi"
]

vid_paths = os.listdir('./Videos/')

print(''), print(colored('...','white')), print(''), print('Choose video'), print('')
terminal_menu = TerminalMenu(vid_paths)
choice_index = terminal_menu.show()
vid_path = f'./Videos/{vid_paths[choice_index]}'
print(f'{vid_paths[choice_index]}'), print('')

########## (Subsection) Set distance detection

print('Distance detection?'), print('')
terminal_menu = TerminalMenu(["True", "False"])
choice_index = terminal_menu.show()
if choice_index == 0:
    distance_detection = True
else:
    distance_detection = False
print(f'{distance_detection}')

#################### (Section) Functions

########## (Subsection) Perspective clicking function

def get_mouse_points(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y
        cv2.circle(frame, (x, y), int(frame_w/100), material.red(shade=50), -1)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        print("Point marked")
        print(x,y)
    # mouse_pts = [(462, 25), (9, 232), (836, 65), (695, 395), (399, 240), (399, 204), (616, 338)]
        
########## (Subsection) Heatmap function

def heatmap(dd):
    for violation in range(len(dd[0])):
        violator1 = dd[0][violation]
        violator2 = dd[1][violation]
        mid_point_x1 = int(
            (pedestrian_boxes[violator1][1] * frame_w + pedestrian_boxes[violator1][3] * frame_w) / 2
        )
        mid_point_y1 = int(
            (pedestrian_boxes[violator1][0] * frame_h + pedestrian_boxes[violator1][2] * frame_h) / 2
        )
        mid_point_x2 = int(
            (pedestrian_boxes[violator2][1] * frame_w + pedestrian_boxes[violator2][3] * frame_w) / 2
        )
        mid_point_y2 = int(
            (pedestrian_boxes[violator2][0] * frame_h + pedestrian_boxes[violator2][2] * frame_h) / 2
        )
        violation_pt_x = mid_point_x1 - (mid_point_x1 - mid_point_x2)/2
        violation_pt_y = mid_point_y1 - (mid_point_y1 - mid_point_y2)/2
        interval = 10
        width_classification = int(np.floor(interval*violation_pt_x/frame_w))
        height_classification = int(np.floor(interval*violation_pt_y/frame_h))
        heatmap_matrix[height_classification, width_classification] += 1
        
    return heatmap_matrix

########## (Subsection) Plot points function

def plot_points_on_bird_eye_view(frame, pedestrian_boxes, M, scale_w, scale_h,d_thresh,heatmap_matrix, bird_height, bird_width):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    ########## (Subsubsection)
    node_radius = int(frame_w/100)
    node_color = material.gray(shade=50)
    node_thickness = -1
    
    background_color = material.gray(shade=90)
    
    hoop_radius = int(d_thresh)
    hoop_color = material.gray(shade=50)
    hoop_thickness = int(node_radius/2)

    violation_color = material.purple(shade=50)

    line_thickness = 5
    ##########################

    blank_image = np.zeros(
        (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
    )
    blank_image[:] = background_color
    warped_pts = []
    
    for i in range(len(pedestrian_boxes)):

        mid_point_x = int(
            (pedestrian_boxes[i][1] * frame_w + pedestrian_boxes[i][3] * frame_w) / 2
        )
        mid_point_y = int(
            (pedestrian_boxes[i][0] * frame_h + pedestrian_boxes[i][2] * frame_h) / 2
        )

        pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
        warped_pt = cv2.perspectiveTransform(pts, M)[0][0]
        warped_pt_scaled = [int(warped_pt[0] * scale_w), int(warped_pt[1] * scale_h)]

        warped_pts.append(warped_pt_scaled)

        bird_image = cv2.circle(
            blank_image,
            (warped_pt_scaled[0], warped_pt_scaled[1]),
            hoop_radius,
            hoop_color,
            hoop_thickness,
        )

        bird_image = cv2.circle(
            blank_image,
            (warped_pt_scaled[0], warped_pt_scaled[1]),
            node_radius,
            node_color,
            node_thickness,
        )

        p = np.array(warped_pts)
        dist_condensed = pdist(p)
        dist = squareform(dist_condensed)
        dist_triu = np.triu(dist)

        pairs = len(dist_triu)

        dd = np.where((dist_triu < d_thresh) & (dist_triu > 0))
        num_violations = len(dd[0])

        if len(dd[0]) > 0:

            for node in range(len(dd[0])):

                bird_image = cv2.circle(
                    blank_image,
                    (warped_pts[dd[0][node]][0], warped_pts[dd[0][node]][1]),
                    node_radius,
                    violation_color,
                    node_thickness,
                )

                bird_image = cv2.circle(
                    blank_image,
                    (warped_pts[dd[1][node]][0], warped_pts[dd[1][node]][1]),
                    node_radius,
                    violation_color,
                    node_thickness,
                )

                cv2.line(
                    bird_image,
                    (warped_pts[dd[0][node]][0], warped_pts[dd[0][node]][1]),
                    (warped_pts[dd[1][node]][0], warped_pts[dd[1][node]][1]),
                    violation_color,
                    line_thickness,
                )

                warped_pt1 = np.array([[[warped_pts[dd[0][node]][0], warped_pts[dd[0][node]][1]]]], dtype="float32")
                warped_pt2 = np.array([[[warped_pts[dd[1][node]][0], warped_pts[dd[1][node]][1]]]], dtype="float32")

                original_pt1 = cv2.perspectiveTransform(warped_pt1, np.linalg.inv(M))[0][0]
                original_pt2 = cv2.perspectiveTransform(warped_pt2, np.linalg.inv(M))[0][0]

                cv2.circle(
                    frame,
                    (int(original_pt1[0]), int(original_pt1[1])),
                    node_radius,
                    violation_color,
                    node_thickness,
                )

                cv2.circle(
                    frame,
                    (int(original_pt2[0]), int(original_pt2[1])),
                    node_radius,
                    violation_color,
                    node_thickness,
                )

                cv2.line(
                    frame,
                    (int(original_pt1[0]), int(original_pt1[1])),
                    (int(original_pt2[0]), int(original_pt2[1])),
                    violation_color,
                    line_thickness,
                )

                violation_pt_x = original_pt1[0] - (original_pt1[0] - original_pt2[0])/2
                violation_pt_y = original_pt1[1] - (original_pt1[1] - original_pt2[1])/2

                interval = 10

                width_classification = int(np.floor(interval*violation_pt_x/frame_w))
                height_classification = int(np.floor(interval*violation_pt_y/frame_h))

    cv2.imshow('Person recognition', frame)
    cv2.imshow("Bird's-eye view", bird_image)
    cv2.moveWindow("Bird's-eye view", int(screen_width/2), 0)
    cv2.waitKey(1)

    return warped_pts, bird_image, pairs, num_violations, dd, heatmap_matrix, frame

########## (Subsection) Perspective function

def get_camera_perspective(img, src_points):
    IMAGE_H = img.shape[0]
    IMAGE_W = img.shape[1]
    src = np.float32(np.array(src_points))
    dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return M, M_inv

########## (Subsection) Grab screen dimensions

screen_width = Tk().winfo_screenwidth()
screen_height = Tk().winfo_screenheight()

########## (Subsection) Start timer

start_time = datetime.now()
print(''), print(colored('...','white')), print(''), print('Started at', start_time.strftime("%H:%M:%S"))
vid_cap = cv2.VideoCapture(vid_path)
vid_fps = vid_cap.get(cv2.CAP_PROP_FPS)

########## (Subsection) Print video information

print(''), print(colored('...','white')), print(''), print(colored('Checkpoint', 'blue'),'Initialisation')
print(''), print(colored('...','white')), print(''), print('Path: {}'.format(vid_path)), print('Width: {} px'.format(int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))), print('Height: {} px'.format(int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))), print('Framerate: {} fps'.format(vid_fps)), print('Duration: {} s'.format(round(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)/vid_fps,2))), print('Frames: {}'.format(int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))))

########## (Subsection) Choose clip region

print(''), print('...'), print(''), print('Clip start?'), print('')
clip_start_s = float(input('Insert float between (0) s and ({}) s '.format(round(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)/vid_fps,2))))
print(''), print('Clip end?'), print('')
clip_end_s = float(input('Insert float between ({}) s and ({}) s '.format(clip_start_s, round(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)/vid_fps,2))))

clip_start = int(clip_start_s * vid_fps)
clip_end = int(clip_end_s * vid_fps)

clip_duration = clip_end - clip_start + 1

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

# height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# output_movie = cv2.VideoWriter("Pedestrian_detect.avi", fourcc, vid_fps, (width, height))
# bird_movie = cv2.VideoWriter(
#     "Pedestrian_bird.avi", fourcc, vid_fps, (int(width * scale_w), int(height * scale_h))
# )

#################### (Section) Start

for frame_idx in range(clip_start, clip_end + 1):

    frame_num += 1

    if frame_idx > clip_end:
        break

    print(''), print(colored('...','white')), print(''), print(colored('New frame', 'green'),f'{frame_idx} ({clip_start} → {clip_end}) ({frame_num} of {clip_duration}) ({int(100*frame_num/clip_duration)}%)')

    vid_cap.set(1, frame_idx)
    (success, frame) = vid_cap.read()

    (frame_h, frame_w) = frame.shape[:2]
    frame = cv2.resize(frame, (0, 0), fx=0.5*screen_width/frame_w, fy=0.5*screen_width/frame_w)
    (frame_h, frame_w) = frame.shape[:2]

    if frame_idx == clip_start:

        if distance_detection == True:

            cv2.imwrite("./Export/frame.png", frame)

            #################### (Section) Perspective
            
            print(''), print(colored('...','white')), print(''), print(colored('Checkpoint', 'blue'),'Perspective')
            print(''),print(colored('...','white')),print(''),print('Mark (Top left) → (Bottom left) → (Top right) → (Bottom right)'),print(''),print(colored('...','white')),print('')
            
            ########## (Subsection) Ask user to mark parallel points and two points 6 feet apart. Order bl, br, tr, tl, p1, p2

            while True:
                cv2.imshow("Perspective", frame)
                cv2.waitKey(1)
                if len(mouse_pts) == 7:
                    cv2.destroyWindow("Perspective")
                    break
                first_frame_display = False
            four_points = mouse_pts

            ########## (Subsection) Get perspective

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

    #################### (Section) Object detection

    obj_start_time = []
    obj_end_time = []
    vio_start_time = []
    vio_end_time = []

    obj_start_time.append(time.time())

    confid = 0.1
    thresh = 0.5

    # wgt_path = "./Yolo/yolov3_brunotimo_5000.weights"
    # cfg_path = "./Yolo/yolov3_brunotimo.cfg"
    
    # wgt_path = "./Yolo/yolov3_brunotimo_10000.weights"
    # cfg_path = "./Yolo/yolov3_brunotimo.cfg"

    # wgt_path = "./Yolo/yolov3_brunotimo_44000.weights"
    # cfg_path = "./Yolo/yolov3_brunotimo.cfg"
    
    # wgt_path = "./Yolo/yolov3_brunotimo_975.weights"
    # cfg_path = "./Yolo/yolov3_brunotimo.cfg"
    
    wgt_path = "./Yolo/yolov3.weights"
    cfg_path = "./Yolo/yolov3.cfg"

    # wgt_path = "./Yolo/yolo-inria_10000.weights"
    # cfg_path = "./Yolo/yolo-inria.cfg"

    labelsPath = "./Yolo/coco.names"

    net = cv2.dnn.readNetFromDarknet(cfg_path, wgt_path)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    labels = open(labelsPath).read().strip().split("\n")

    frame_resized = cv2.resize(frame, (416, 416))                                                   # Scale image by dividing by 255. YoloV3 needs input size (416, 416)
    blob = cv2.dnn.blobFromImage(
        frame_resized, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    blobb = blob.reshape(blob.shape[2], blob.shape[3], 3)

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

    centers = [] 

    if len(idxs) > 0:
        boxes = np.array(boxes)[idxs.flatten()]
        boxes2 = np.array(boxes2)[idxs.flatten()]
        boxes_norm = np.array(boxes_norm)[idxs.flatten()]
        boxes_norm2 = np.array(boxes_norm2)[idxs.flatten()]
        confidences = np.array(confidences)[idxs.flatten()]
        classIDs = np.array(classIDs)[idxs.flatten()]

        print(''),print(colored('...','white')),print(''),print('Confidences:', [round(num, 2) for num in confidences])

        status = []
        idf = idxs.flatten()
        close_pair = []
        s_close_pair = []
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
    # cv2.imwrite(f"./frame.png", frame)
    cv2.waitKey(1)
        
    print('Centers:', centers)

    if distance_detection == True:

        pts = np.array(
            [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32
        )
        cv2.polylines(frame, [pts], True, (0, 255, 255), thickness=4)

    pedestrian_boxes = boxes_norm2
    num_pedestrians = len(boxes_norm2)

    print(''), print(colored('...','white')), print(''), print(colored('Checkpoint', 'blue'),'Object detection')

    obj_end_time.append(time.time())

    if distance_detection == True:

        vio_start_time.append(time.time())

    ########## (Subsection) Detect person and bounding boxes using DNN

        if len(pedestrian_boxes) > 0:

            print(''), print(colored('...','white')), print(''), print(colored('Checkpoint', 'blue'),'Social distancing'), print(''), print(colored('...','white')), print('')
            
            warped_pts, bird_image, pairs, num_violations, dd, heatmap_matrix, frame = plot_points_on_bird_eye_view(
                frame, pedestrian_boxes, M, scale_w, scale_h,d_thresh, heatmap_matrix, bird_height, bird_width
            )

            num_violations_cumulative += num_violations
            num_pedestrians_cumulative += num_pedestrians

            heatmap(dd)

            print(f'Pedestrians: {num_pedestrians} ({num_pedestrians_cumulative})')
            print(f'Pairs: {pairs}')
            print(f'Violating pairs: {dd}')
            print(f'Violations: {num_violations} ({num_violations_cumulative})')

            print(''), print(colored('...','white')), print(''), print(colored('Checkpoint', 'blue'),'Social distancing performance'), print(''), print(colored('...','white')), print('')
            print(f'Frames: {frame_num}')
            print(f'Violations (average): {round(num_violations_cumulative/frame_num,1)}')
            print(f'Pedestrians (average): {round(num_pedestrians_cumulative/frame_num,1)}')

            if ((frame_num % 10 == 0) and (frame_num > 0)):
                print('print')
                fig = plt.figure(figsize=(1, 1))
                dpi = fig.get_dpi()
                plt.close()
                fig = plt.figure(figsize=(frame_w/float(dpi),frame_h/float(dpi)))
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                seaborn.heatmap(heatmap_matrix, cbar = False)
                plt.savefig("./Export/heatmap.png")

        # cv2.imwrite(f"./Export/3D_{frame_idx}.png", frame)
        # cv2.imwrite(f"./Export/2D_{frame_idx}.png", bird_image)

        vio_end_time.append(time.time())


print(''), print(colored('...','white')), print('')
print(f'Object detection took an average of ({np.sum([x - y for x, y in zip(obj_end_time, obj_start_time)])/len(obj_end_time)}) s'), print('')
print(f'Violence detection took an average of ({np.sum([x - y for x, y in zip(vio_end_time, vio_start_time)])/len(vio_end_time)}) s')

end_time = datetime.now()
print(''), print(colored('...','white')), print(''), print(f'Ended at {end_time.strftime("%H:%M:%S")} ({end_time-start_time})'),print(''), print(colored('...','white')), print('')

#################### (Section) Results

########## (Subsection) Heatmap

fig = plt.figure(figsize=(1, 1))
dpi = fig.get_dpi()
plt.close()
fig = plt.figure(figsize=(frame_w/float(dpi),frame_h/float(dpi)))
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
seaborn.heatmap(heatmap_matrix, cbar = False)
plt.savefig("./Export/heatmap.png")

combined = cv2.addWeighted(cv2.imread("./Export/heatmap.png"), 0.5, cv2.imread("./Export/frame.png"), 0.5, 0)

cv2.imwrite("./Export/combined.png", combined)
cv2.imshow('Heatmap', combined)
cv2.waitKey(0)