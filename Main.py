########################################
########################################
########################################
########################################
# Bruno Martens & Timo Vijn
# Social Distancing
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

#################### (Section) Figures

########## (Subsection) Define clicking function

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

########## (Subsection) Define point function

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

                # cv2.waitKey(0)
                
                # width_classification1 = int(np.floor(interval*original_pt1[0]/frame_w))
                # height_classification1 = int(np.floor(interval*original_pt1[1]/frame_h))

                # width_classification2 = int(np.floor(interval*original_pt2[0]/frame_w))
                # height_classification2 = int(np.floor(interval*original_pt2[1]/frame_h))

                # heatmap_matrix[height_classification, width_classification] += 1
                # heatmap_matrix[height_classification2, width_classification2] += 1
                # print(heatmap_matrix)
    cv2.imshow('Person recognition', frame)
    cv2.imshow("Bird's-eye view", bird_image)
    cv2.moveWindow("Bird's-eye view", int(screen_width/2), 0)
    cv2.waitKey(1)

    return warped_pts, bird_image, pairs, num_violations, dd, heatmap_matrix, frame

def get_camera_perspective(img, src_points):
    IMAGE_H = img.shape[0]
    IMAGE_W = img.shape[1]
    src = np.float32(np.array(src_points))
    dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return M, M_inv

# def put_text(frame, text, text_offset_y=25):
#     font_scale = 0.8
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     rectangle_bgr = (35, 35, 35)
#     (text_width, text_height) = cv2.getTextSize(
#         text, font, fontScale=font_scale, thickness=1
#     )[0]
#     # set the text start position
#     text_offset_x = frame.shape[1] - 400
#     # make the coords of the box with a small padding of two pixels
#     box_coords = (
#         (text_offset_x, text_offset_y + 5),
#         (text_offset_x + text_width + 2, text_offset_y - text_height - 2),
#     )
#     frame = cv2.rectangle(
#         frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED
#     )
#     frame = cv2.putText(
#         frame,
#         text,
#         (text_offset_x, text_offset_y),
#         font,
#         fontScale=font_scale,
#         color=(255, 255, 255),
#         thickness=1,
#     )

#     return frame, 2 * text_height + text_offset_y


# def calculate_stay_at_home_index(total_pedestrians_detected, frame_num, fps):
#     normally_people = 10
#     pedestrian_per_sec = np.round(total_pedestrians_detected / frame_num, 1)
#     sh_index = 1 - pedestrian_per_sec / normally_people
#     return pedestrian_per_sec, sh_index


# def plot_pedestrian_boxes_on_image(frame, pedestrian_boxes):
#     frame_h = frame.shape[0]
#     frame_w = frame.shape[1]
#     thickness = 2
#     # node_color = (192, 133, 156)
#     node_color = (160, 48, 112)
#     # color_10 = (80, 172, 110)

#     for i in range(len(pedestrian_boxes)):
#         pt1 = (
#             int(pedestrian_boxes[i][1] * frame_w),
#             int(pedestrian_boxes[i][0] * frame_h),
#         )
#         pt2 = (
#             int(pedestrian_boxes[i][3] * frame_w),
#             int(pedestrian_boxes[i][2] * frame_h),
#         )

#         frame_with_boxes = cv2.rectangle(frame, pt1, pt2, node_color, thickness)


#     return frame_with_boxes

########## (Subsection) Grab screen dimensions

screen_width = Tk().winfo_screenwidth()
screen_height = Tk().winfo_screenheight()

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

clip_start_s = 5
clip_end_s = 6

clip_start = int(clip_start_s * vid_fps)
clip_end = int(clip_end_s * vid_fps)
clip_duration = clip_end - clip_start + 1

########## (Subsection) Print video information

print(''), print(colored('...','white')), print(''), print(colored('Checkpoint', 'blue'),'Initialisation')
print(''), print(colored('...','white')), print(''), print('Path: {}'.format(vid_path)), print('Width: {} px'.format(int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))), print('Height: {} px'.format(int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))), print('Framerate: {} fps'.format(vid_fps)), print('Duration: {} s'.format(round(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)/vid_fps,2))), print('Frames: {}'.format(int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))))

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

        cv2.imwrite("./Export/frame.png", frame)

        #################### (Section) Perspective
        
        print(''), print(colored('...','white')), print(''), print(colored('Checkpoint', 'blue'),'Perspective')
        print(''),print(colored('...','white')),print(''),print('Mark (Bottom left) → (Bottom right) → (Top left) → (Top right)'),print(''),print(colored('...','white')),print('')
        
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
    cv2.polylines(frame, [pts], True, (0, 255, 255), thickness=4)

    pedestrian_boxes = boxes_norm2
    num_pedestrians = len(boxes_norm2)

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
        print(f'Violations: {round(num_violations_cumulative/frame_num,1)}')
        print(f'Pedestrians: {round(num_pedestrians_cumulative/frame_num,1)}')

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

    cv2.imwrite(f"./Export/3D_{frame_idx}.png", frame)
    cv2.imwrite(f"./Export/2D_{frame_idx}.png", bird_image)

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