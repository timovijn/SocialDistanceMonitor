# Copy of Social Distance Monitor

---

---

# Introduction

With the rise of the COVID-19 pandemic, society calls for innovative solutions to solve the emerging problems and aid humanity in dealing with the new life standards and regulations. One of these regulations is that of social distancing, where civilians are asked to not get any closer to each other than a by government predefined distance. As the goal is to reduce the total amount of COVID-19 contaminations, the need for an application that can show where and how often the regulations are violated is high. This blog post comprises of a tool, the Social Distance Monitor (SDM), that evaluates recordings of surveillance cameras detects all people present and then presents the total amount of people that violate the taken measures and the locations where the majority of these violations occur. This way, authorities can adjust the spatial planning accordingly, with the goal to reduce the chance that COVID-19 is spread within their region. For this program to be useful, it should adhere to the following requirements:

- The execution time of the algorithm on a single frame should be lower than 0.5 seconds on professional hardware.
- The accuracy of people recognition should be higher than 80%.
- The accuracy of distance detection should be higher than 80%.
- It should be robust in the sense that the program handles a variety of crowds, perspectives, resolutions, lighting conditions and environmental differences. To measure this, 10 clips containing these different conditions have been carefully selected as presented in the section on [Input Data](https://www.notion.so/timovijn/Social-Distance-Monitor-ba55fb2f74b44aed91a729e04bdeb00e#05802bc1d33c4dbaa92b29d6a159a609).
- The operating time should be limited to a 1 time setup with a duration less than 10 minutes.
- The tool should be user-friendly in the sense that it easy to operate and also outputs a variety of data.

The ‘Social Distance Monitor’ was developed during times of the COVID-19 epidemic, however its use is much more versatile than crime prevention and law enforcement only. For example, one can use it to adjust infrastructure or urban planning based on pedestrian flows and measure the effect of events on specific parts of their domain.

*The Social Distance Monitor is publicly available on [Github](https://github.com/timovijn/SocialDistanceMonitor).*

# Related Work

To go from idea to a working tool, inspiration was taken from existing variants and widely used techniques were used to execute specific phases in the program such as the YOLO neural network to detect pedestrians. These other implementations and underlying principles will be set out below.

### Underlying Principles

- To calculate the distance between two people, the program first need to know where the persons are located. Therefore the tool should be able to identify objects, or in this case persons, from video footage. Object detection is a wide area of research within the deep learning community and several architectures are widely distributed and used that can serve in this purpose. To classify humans from video footage in our tool, a trained neural network called [You Only Look Once (YOLOv3)](https://arxiv.org/pdf/1506.02640v5.pdf). YOLO is a real-time object detection system. The single neural network it consists of looks at the image once, divides it into regions and predicts bounding boxes and probabilities for each region. Finally, the bounding boxes are weighted by the predicted probabilities. The pretrained weights available are trained using the [COCO-dataset](http://cocodataset.org/#home), which consists of images of common objects (including persons/pedestrians).
- As the problem introduced lies within the field of computer vision, this project makes use of [OpenCV](https://opencv.org/). The cv2 module is an renowned image and video processing library providing many capabilities. In this project, mainly the deep neural network modules, as well as the video processing tools were used.

### Other Implementations

- Landing AI, a famous start-up in the field of Artificial Intelligence, built a [social distancing detector](https://landing.ai/landing-ai-creates-an-ai-tool-to-help-customers-monitor-social-distancing-in-the-workplace/). The detector works roughly the same as the one presented in this blog post, as they use the video surveillance input, detect pedestrians and evaluate the social distancing violations from a bird's-eye perspective. Where the algorithms differ is in the sense of object detection. Landing AI uses Faster R-CNN, where our program uses YOLOv3, which will be explained later. Landing AI’s tool does only show the violations as opposed by our tool, where the locations of occurrence are also given as output. This is where our program provides novel insights.
- Aqeel Anwar also came up with a [likewise tool](https://towardsdatascience.com/monitoring-social-distancing-using-ai-c5b81da44c9f), again using the bird's eye perspective to measure the violations of social distancing. His implementation of transforming the perspective to bird's eye view was used as basis for the tool in this post. A limitation in his work is the way the distance between identified persons is calculated. This problem is addressed in the method section.
- Lastly, a [social-distancing-analyser](https://github.com/Ank-Cha/Social-Distancing-Analyser-COVID-19) by Ankush Chaudhari was used as inspiration for our algorithm. The way the YOLO neural net in implemented is equal to that of this blog post. This implementation does not make use of bird's eye view and thus makes up for a interesting comparison to the other implementations including that proposed this post.

# Methods

Methods are discussed that are used for the Social Distance Monitor and for training a custom YOLO network for person detection.

## Social Distance Monitor

The algorithm of the Social Distance Monitor consists of a number of constituent parts (phases) that will be explained separately in an input-to-output manner.

The overarching goal of the algorithm is to provide the user with insightful indicators indicating the amount and location of social distance violations for an input of pedestrian video footage. When we descend into a more detailed level, the algorithm’s phases shown in the table below can be considered separately. The algorithm progresses through these phases for every frame of a video separately and linearly.

| Input | Phase | Output |
|-------------------------|-------------------------|---------------------------------|
| Raw video footage → | (1) Video processing | → Processed video frame |
| Processed video frame → | (2) Object detection | → Person locations (3D) |
| Person locations (3D) → | (3) Perspective change | → Person locations (2D) |
| Person locations (2D) → | (4) Violation detection | → Violations |
| Violations → | (5) Indicators | → Indicators and visualisations |

### (Phase 1) **Video processing**

To obtain (output) a processed video frame from (input) raw video footage, the algorithm contains a phase for video processing.

This step in the algorithm is a straightforward one which starts by loading a file of any video format for the timestamps set by the user, and loading a single frame from this video. Every input video can have its own dimensions. Output is standardised such that frames are always shown in the same horizontal dimension. Furthermore, a frame is always resized to 416x416 resolution before continuing to the next step.

Motivated by our requirement of user-friendliness, the algorithm provides the user with information about the selected video and clip.

```python
Started at 12:56:50

...

Path: ./Videos/TownCentreXVID.avi
Width: 1920 px
Height: 1080 px
Framerate: 25.0 fps
Duration: 300.0 s
Frames: 7500
```

### (Phase 2) **Object detection**

To obtain (output) person locations in 3D from (input) a processed video frame, the algorithm contains a phase for object detection.

As stated before, object detection is implemented through the YOLO object detection system. The three required files are stored locally to allow configuration before being loaded for the algorithm to use. The processed frame from one phase earlier is further processed by `blob = cv2.dnn.blobFromImage()`. In our case, this function is used to create a 4-dimensional blob from the frame with swapped Blue and Red channels.

The artificial network is initialised by `net = cv2.dnn.readNetFromDarknet()` which takes the weight file and configuration file as input. The input to this network is then set by `net.setInput()` and layer outputs are obtained by `net.forward()`. This output is used to determine bounding boxes of the objects that have been detected to be of class ‘person’ with a confidence value of at least a user set value. Bounding boxes are constructed depending on the confidence and threshold, which are both set to 0.5.

### (Phase 3) Perspective change

To obtain (output) person locations in 2D from (input) person locations in 3D, the algorithm contains a phase for perspective change.

The phase to change perspective is to obtain a top view (bird’s eye view) of the scene. This phase is motivated by the fact that – given that this view is a realistic representation of reality – social distance violations can be determined easily by circles. Furthermore, some users may find this view to be more insightful, and it may help to create additional visualisations at a later stage.

At the foundation of this phase is `cv2.perspectiveTransform()`, a function that requires two equal-sized source and destination arrays to determine the transformation matrix. The source or “to-be-transformed’ array is obtained manually. At the start of the algorithm, there is a popup that directs the user to mark four points that indicate the perspective of the scene. These points are then used as source array. After obtaining the transformation matrix, it is straightforward to map the centre point of the bounding box of a person from the original 3D view to a “warped” centre point on the 2D bird’s eye view. Note that the people walking in the length of the street from nearer to farther from the camera, walk from right to left on the bird’s eye view.

![Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/video.gif](Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/video.gif)

Original clip from Oxford Town Centre (left) and corresponding bird’s eye view (right)

![MovingOutput](https://github.com/timovijn/SocialDistanceMonitor/blob/master/Blog/Images/MovingOutput.gif?raw=true)
*Original clip from Oxford Town Centre (left) and corresponding bird’s eye view (right)*

### (Phase 4) Violation detection

To obtain (output) violations from (input) person locations in 2D, the algorithm contains a phase for violation detection.

Because of the bird’s eye view from the previous phase, the phase of violation detection contains only basis mathematical operations. A relationship matrix is computed that contains the distance from every point to all other points. All relationships in the upper triangular part of this symmetric matrix that do not exceed the given social distance are marked as being a violation and a line is drawn between the two people that are in violation on both 2D and 3D views.

### (Phase 5) Indicators

To obtain (output) indicators from (input) violations, the algorithm contains a phase for indicators.

One of the most informative visualisations is that of the Violation Heatmap, which exactly fits the original scene and can thus be overlaid as demonstrated by the right image below. To establish this heatmap, the centre-point of violation – that is, the point that is exactly in between two people that are in violation – is recorded as the violation’s location.

| [![Heatmap](https://github.com/timovijn/SocialDistanceMonitor/blob/master/Blog/Images/MovingOutput.gif?raw=true)](Violation Heatmap)  | [![HeatmapOverlay](https://github.com/timovijn/SocialDistanceMonitor/blob/master/Blog/Images/HeatmapOverlay.png?raw=true)](http://audioblocks.com) |

![Heatmap](https://github.com/timovijn/SocialDistanceMonitor/blob/master/Blog/Images/Heatmap.png?raw=true) ![HeatmapOverlay](https://github.com/timovijn/SocialDistanceMonitor/blob/master/Blog/Images/HeatmapOverlay.png?raw=true)
*Violation Heatmap (left) and Violation Heatmap overlaying a frame from the Oxford Town Centre (right)*

Besides the Violation Heatmap, the algorithm returns (a) the amount of pedestrians, and (b) the amount of violations for every frame as well as cumulative, and also (c) which pairs are in violation for every frame. Furthermore, the algorithm keeps track of the average social distancing performance in terms of average violations and average pedestrians. On the basis of this data, a magnitude of interesting output could be constructed that complements the Violation Heatmap.

```python
New frame 575 (500 → 575) (76 of 76) (100%)

...

Pedestrians: 21 (1648)
Pairs: 21
Violating pairs: (array([ 6, 10, 10, 10, 11, 11, 15, 17]), array([ 9, 11, 15, 19, 15, 19, 19, 18]))
Violations: 8 (690)

...

Frames: 76
Violations (average): 9.1
Pedestrians (average): 21.7
```

Most choices that were made during the development of the Social Distance Monitor are motivated by the requirement of user-friendliness. Although similar tools exist, see the section on [Other Implementations](https://www.notion.so/timovijn/Social-Distance-Monitor-ba55fb2f74b44aed91a729e04bdeb00e#8226a400f4ac4215beaed8da44966aa1), the Social Distance Monitor has been developed to be closer to what can actually be used.

At the front-end the process of choosing an input is straightforward and works for a wide variety of video types and resolutions. A selection menu in the terminal helps with choosing video source and clip. When the algorithm finishes processing the final frame from the selected clip, the heatmap comes up and shows problematic areas.

At the back-end, it has been chosen to implement the violence detection phase after the object detection phase within every iteration of a frame; when a network is used that is sufficiently lightweight, such that frames are processed at at least ~30 fps, the complete process can then run on camera footage in real-time. Whether this is required for future applications should be considered on a case-to-case basis. On our setup (3,1 GHz Dual-Core Intel Core i5), the execution time of the object detection phase (including visualisations) was 1.30 seconds on average, significantly higher than that of the violation detection phase (including visualisations) which was only 57.0 milliseconds on average. What limits execution speed is clearly the object detection phase. More professional setups should have no such problems as they have been successful to run [YOLOv3 object detection at 30 fps](https://pjreddie.com/darknet/yolo/), although accuracy has to be taken into consideration as well and could provide to be another limiting factor.

## Training YOLO for person detection (only)

### Preparing training on a custom dataset

As discussed in the [related work section](https://www.notion.so/Social-Distance-Monitor-ba55fb2f74b44aed91a729e04bdeb00e#abfa7ea204314ff9b083b3d50ed53d8c), in order to construct a working social distance monitor, the tool should be able to detect persons based on video footage.  The choice for YOLOv3 was easy based on the requirements that were set for this tool. The detection system should work fast as well as accurate. As YOLO is the state-of-the-art, real time object detection system it outperforms likewise systems as RetinaNet (both the 50 and 101 equivalent) and Fast(er) RCNN in these requirements. This comes at the cost of detection accuracy of really small objects within the image considered. As we observed humans in general video footage to be relatively large, the choice for YOLO could still be justified. YOLOv3 was chosen as it complies well with the openCV module used in the rest of the proposed tool.

The next step was to think of a YOLO architecture and corresponding set of data that, when trained, would best serve the purpose of the Social Distance Monitor.  The idea was to train the YOLO network solely for the purpose of human detection, instead of on all objects present in the COCO-dataset as in the training process of the pre-trained weights. The reasoning for this was that this would result in a faster training process and a possibly a better accuracy or match the accuracy of the pretained weights within a shorter amount of epochs.

To execute this idea, the first step was to choose a dataset consisting of images of persons, with corresponding labels. The authors of the blogpost selected this dataset with great consideration taking into account the perspective, resolutions, and lighting conditions of the image present within the dataset. Secondly apart from how well the dataset represents the actual applications the architecture will be applied to, the amount of training data was also an important factor in choosing our final dataset. This way three datasets were chosen (presented in order of preference): [Caltech Pedestrian Dataset](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/), [Crowd-Human Dataset](http://www.crowdhuman.org/download.html), [INRIA Pedestrian Dataset](https://dbcollection.readthedocs.io/en/latest/datasets/inria_ped.html). More about these datasets and their corresponding preprocessing phase is covered in the next section.

As part of of the training procedure, [Darknet](https://pjreddie.com/darknet/) was used as open source framework to train our yolo architecture with, since it ensured compatibility with the online training environment [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true). Colab was introduced to be able to train the network within less time due to the capabilities their GPUs provide. 

Now the network, the dataset and the training data are chosen, the way to move forward is to decide the exact configuration of the to be trained YOLO network. Before diving into this, a deeper explanation of the YOLO network is given. 

YOLO consists of a single convolutional network that predict bounding boxes while making use if dimension clusters as anchor boxes. An image is used as input into the network and is being looked at only once (declaring the name You Only Look Once), which actually means that it is propagated throughout the network in a forward fashion. The input image is split in an AxA grid. Within one of these grid segments, B bounding boxes are made. For each of these bounding boxes the probability that it contains an object is determined. Afterwards [non-max suppression](https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c) is applied with the goal to reduce the chance that one object/person is detected multiple times. The output of the network are the recognised persons/objects and their corresponding bounding boxes. Using the sum of squared error loss the classification accuracy of the network is measured [1] [2]. A schematic representation of the network can be found below. The Convolutional Neural Network consists of 53 convolutional layers, which are all followed by Batch Normalization and a Leaky ReLU activation function.

![Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/Screenshot_2020-07-01_at_18.45.25.png](Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/Screenshot_2020-07-01_at_18.45.25.png)

Schematic representation of the YOLOv3 architecture with Darknet 53 as backbone [1].

Now we know how the architecture is built up, one could consider optimising for their specific application. Crafting a self-made architecture must be done with great consideration if one plans to end up with a usable network. To optimise the network for our application we changed the amount of classes to 1, as we only try to detect persons. In order to account for this we need to change the amount of filters before each yolo layer according to the formula (5+classes)*3 = 18. Width and height (to which the CNN will resize your input image was increased from 416*416 to 608*608. This will reduce the training speed, but increase the accuracy as the image propagated throughout the convolutional layers will contain more features. 

### Training

Having completed all steps to prepare for training the network, it is time to initiate the training process. For this, the following [Colab Notebook](https://colab.research.google.com/drive/1amwKEOPiutRA5OI_AbrvgTnYNp_t9h0i#scrollTo=qhRawiQTlqE8) (which works the same as a Jupyter Notebook (.ipynb)) was used. This notebook will load all required data from the dataset (which should be uploaded to a Google Drive account first. The next step is to compile with Darknet. Following, all files are converted towards the right format so no errors occur during training. Finally, the training process is started with a training command which inputs the correct files into the network. This way, one can also pickup training from a later point, with a saved weight file. 

### Results

Having experienced various complications throughout the training phase, we eventually decided to move on towards the INRIA pedestrian dataset (treated in the conclusion). Using this dataset, the architecture specified above was tested for a more than 120000 batches (>11.000 epochs). Manually, the different weight files originating from this testing process were tested against both the training as well as the test set. As overfitting behaviour was observed for the later weight files, the training phase was stopped. If one looks at the loss curve, a good generalising behaviour can be observed. During the start of the training process two big spikes and one smaller spike could be observed in the curve. The definite reason for this is unclear, but a potential cause is in the setup of the dataset, where not all persons present in the dataset were labelled. When presented images containing many people (that are not labelled) in close fashion, a lot of misclassifications are made in a short time, resulting in a higher loss.

![Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/avg_loss-1.png](Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/avg_loss-1.png)

Loss curve for the training process for the self-trained version of the YOLOv3 network on the INRIA pedestrian dataset.

Below one can find images from the training and test set and how our self-trained network classified them.

![Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/frame.png](Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/frame.png)

Classification of a random sample from the train set (YOLOv3 on INRIA).

![Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/frame%201.png](Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/frame%201.png)

Classification of a random sample from the test set (YOLOv3 on INRIA).

# Data & Experiments

## Data

While working on the Social Distance Monitor a large variety of data was used. For the different phases, different datasets were used to be able to get an accurate and working tool. In this section all kinds of data used for the project are set out. 

### Input Data

In order to build the social distance monitor a variety of data was used. First of all, the input of the program is a video (mostly from surveillance cameras). For these videos as input data, footage from a variety of video datasets was used. The goal was to select a range of videos with differing characteristics such as lighting, amount of pedestrians, perspectives and surrounding objects in the environment. The selection of these videos was done manually with careful consideration based on the previously mentioned criteria. Footage from the following datasets was used: [Oxford Town Centre](https://megapixels.cc/oxford_town_centre/), [CAVIAR,](http://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) [Virat Video Dataset](https://viratdata.org/), [EPFL](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/) and an additional test video was provided by [BriefCam](https://www.youtube.com/watch?v=aUdKzb4LGJI). 

### Data for person detection

As this is the data used as input of training the architecture, it is not used in the social distance monitor itself. For person recognition, this can be split up in two parts, being the data used for the pre-trained weights of the YOLO network and the data used for the self-trained YOLO network. 

The [COCO-dataset](http://cocodataset.org/) was used to train and craft the original YOLO network. This dataset, containing over 200,000 labeled images, contains 80 of the most common objects and their corresponding labels. It forms the basis for the pre-trained YOLOv3 variant. 

For the custom self-trained variant of the YOLO network, the [Caltech Pedestrian Dataset](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/) was introduced. This is the largest pedestrian dataset available. It consists of 250.000 individual frames. From the total dataset of 70GB, a subset was selected. This way we ensured that we could train our network within the limits of Google Colab. We trained this network for an amount 40.000 epochs, however the results were far from promising (see conclusion for more detail).

The second dataset selected was the [Crowd-Human Dataset](https://www.crowdhuman.org/). This dataset consists of 15.000 images containing an astonishing amount of 340.000 persons. The training was successfully initialised after preprocessing the dataset but crashed due to an unknown reason at a random location within the first 200 epochs.

Therefore another dataset was added to the process. This dataset has a smaller amount of images, as during the process of the two other datasets, the limits of Google Colab were reached several times. This [INRIA Person Dataset](https://dbcollection.readthedocs.io/en/latest/datasets/inria_ped.html) contains positive and negative samples, i.e images with people present in them and images were no people are observed. The 1.1 GB dataset contains of 1832 images with 614 positive labels in the training set and 741 images with 288 positive labels in the test set. To get a better indication on what kind of images are present in the dataset, one can take a look at the positive samples shown in the previous section.

### Pre-processing

Although we pre-processed three different datasets to train the self-trained version of the YOLO network on, we only state our process of the INRIA dataset here, as we know this process is correct due to the successful training.

To be able to use the INRIA Dataset to train the YOLOv3 network, a few preprocessing steps were performed. Yolo can handle images with varying size as input, so no image resizing needed to be done. The labels however needed to be converted to another format. YOLO uses labels with the following format: <class> <x> <y> <width> <height>. As the dataset provided only the X and Y coordinates of the persons in the images, a [Python file](https://github.com/Zyjacya-In-love/Pedestrian_Detection_YOLOv3_in_INRIA#21-make-yolo-data) was used to convert the labels to the correct format. In the same Github repo, a file could be found that generated the needed train.txt and test.txt files. 

## Performance and robustness experiments

Experiments that give an indication of the performance and robustness of the Social Distance Monitor, most that vary the input video footage, are shortly addressed in terms of description, hypothesis, and result.

### (Experiment a) Decreased resolution

Decrease video to a lower resolution. Run SDM for the original video and resolution-decreased video for the exact same frames.

The hypothesis for this experiment is that the person detection will suffer somewhat from downscaling the video, and more with decreased resolution. Other aspects of the SDM have no further dependency on the resolution of the clip. Social distance identification is thus expected to have decreased accuracy due to incorrect recognition of persons.

For this experiment, the resolution of the Oxford Town Centre video (1920x1080) was reduced to 241 × 136. For the exact same frames for both videos, the average amount of pedestrians in a single frame is 12.6 for the original resolution, and 11.1 (-12%) for the decreased resolution. The frames below quite clearly demonstrate this observation. For the video of decreased resolution, the algorithm detects some people, but clearly misses a person at the shopping windows on the left, two persons at the middle top of the frame, and a person with a bike at the top right of the frame. These persons are all recognised for the video of original resolution. However, the person with the bike is classified as two persons, which demonstrates that even at higher resolutions, recognition is not flawless. The average amount of violations in a single frame is 2.4 for the original resolution, and 1.8 (-25%) for the decreased resolution. A lot of violations occur at the top of the frame, where it becomes more and more difficult to recognise persons at a significantly reduced resolution. A violation occurs between two people, and when one person from this couple is not recognised properly, the violation is not identified. For this reason, the amount of violations is even more sensitive to the reduction of video resolution than the identification of people is. Furthermore, the experiment demonstrates that the heatmap that arises from both videos is similar and leads to a more or less the same conclusion about where the most violations occur. One could thus consider to use videos of decreased resolution when used for the purpose of violation location solely.

```markdown
Pedestrians: 10 (1396)
Pairs: 10
Violating pairs: (array([4, 6]), array([5, 9]))
Violations: 2 (228)

...

Frames: 126
Violations: 1.8
Pedestrians: 11.1
```

![Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/3D_85.png](Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/3D_85.png)

Frame from Oxford Town Centre at 241x136 px

![Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/combined.png](Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/combined.png)

Violation Heatmap for Oxford Town Centre at 241x136 px

```markdown
Pedestrians: 12 (1588)
Pairs: 12
Violating pairs: (array([0, 9]), array([ 7, 11]))
Violations: 2 (307)

...

Frames: 126
Violations: 2.4
Pedestrians: 12.6
```

![Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/3D_85%201.png](Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/3D_85%201.png)

Frame from Oxford Town Centre at 241x136 px

![Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/combined%201.png](Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/combined%201.png)

Violation Heatmap for Oxford Town Centre at 1920x1080 px

### (Experiment b) Different height perspective

Record two videos of same scene, from different height perspectives. Run SDM for both videos.

The hypothesis for this experiment is that the person detection will be exactly the same for both clips, as long as both clips are recorded at a reasonable camera height. The perspective transformation is expected to be of a lower accuracy for the clip that has been recorded at a lower height. Therefore, the social distance identification is expected to have decreased accuracy due to reduced accuracy in the phase of perspective transformation.

To our knowledge, there is no readily available video footage of the same scene at different heights. Luckily our home in Delft had ideal conditions to carry out this experiment. By recording from the second floor and from the third floor at the same time, pedestrian footage at different heights was obtained that will not be portrayed for privacy reasons. From this data it became clear that the hypothesis was mostly correct. Both clips were taken from non-extreme height, and apart from some exceptions, classification was very similar in both cases. Although there was not a staggering difference, the bird’s eye view – and therefore the distance between pedestrians – of the footage recorded from the third floor was slightly more accurate.

### (Experiment c) People farther away

In the same scene, identify at least two groups of people that are more or less at the same distance from each other, but at different distances from the camera.

It is expected that the group of people that are more in the back of the scene (far from the camera) will have a more inaccurate location on the bird’s eye view due to inaccuracies in the method of perspective transformation that become more significant when farther away from the camera.

For this experiment, the SDM is run for a scene from the [Virat Video Dataset](https://viratdata.org/). This scene contains certain frames where two groups of people that are at a different distance from the camera are clearly distinguishable. The group of three is closer to the camera and have been identified as being in violation with one another. The second group of four is farther from the camera, and are also identified as being in violation with one another. Upon inspection of the bird‘s eye view, the  seems to be no significant decrease of accuracy when comparing the second group to the first; neither groups seem to have been mapped flawlessly to the bird’s eye view, but the mapping of both groups is definitely adequate for the purpose of social distance detection.

![Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/Screenshot_2020-07-01_at_20.04.33.png](Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/Screenshot_2020-07-01_at_20.04.33.png)

Original frame from Virat Video Dataset footage (left) and corresponding bird’s eye view (right)

## Shortcomings

As demonstrated in the experiments, the SDM has shortcomings with regard to perspective. The transformation from a 3D scene to a 2D scene introduces inaccuracies that ripple on in the social distance identification. An observation that is very clear when considering the [animated frames above](https://www.notion.so/timovijn/Social-Distance-Monitor-ba55fb2f74b44aed91a729e04bdeb00e#d0f43a21e1a04cc4a11172cc47b56578) is that the points on the bird’s eye view jiggle around quite a lot and some more than others. The reason for this is that the bounding boxes jiggle around as well. When concentrating on a single person for a number of constituent frames, one will probably observe this effect; moving body parts such as legs and arms lead to a slight variation in the centre and size of the bounding boxes for the same person in a scene.

Another shortcoming that becomes more clear when considering the applicability of such a tool on large scale is the fact that the perspective of a scene has to be set manually. This means that for future input, which could possibly or even probably be a lot of different scenes that all have their own perspective, someone would need to manually go through them, which would introduce a lot of inaccuracy in itself.

In terms of the training, it can be said that despite of the promising results that the images in the test set show, the self-trained network overlooks a large amount of pedestrians when deployed on the video surveillance footage. Due to this inaccuracy the Social Distance Monitor works better with the pre-trained weights as input. As mentioned earlier, problems were encountered while training our own YOLO-network for pedestrian detection only. Eventually the training process was performed on a dataset containing a relatively low amount of images. A potential reason for this inaccuracy on the surveillance footage is that either the amount of images in the dataset is to low or the images in the dataset lie to much in the same domain, and thus only images or frames with similar graphical content as in the used dataset will be classified correctly. Upon selecting the datasets these aspects were taken into account and we even tried to account for a variety of resolutions and environmental conditions, but as a lot of difficulties arose during the training phase, another dataset than our initial preference had to be chosen. The images below demonstrate that the self-trained weights perform well on footage recorded from ground level, but have lacking performance on footage recorded at video surveillance level.

![Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/frame%202.png](Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/frame%202.png)

YOLOv3 pre-trained weights (10,000 epochs) on frame from Oxford Town Centre

![Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/frame%203.png](Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/frame%203.png)

YOLOv3 pre-trained weights on frame from CAVIAR

![Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/frame%204.png](Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/frame%204.png)

YOLOv3 self-trained weights (10,000 epochs) on frame from Oxford Town Centre

![Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/frame%205.png](Copy%20of%20Social%20Distance%20Monitor%20ee86ab6fa90d43b19f9d3079463ff5ab/frame%205.png)

YOLOv3 self-trained weights (10,000 epochs) on frame from CAVIAR

# Conclusion

In this section we first cover the evaluation of the Social Distance Monitor, after which we will reflect upon our own learning process and we conclude this post with recommendations for further research.

## Summary

Reflecting upon the Social Distance Monitor in its current form it can be concluded that an fast, accurate, user friendly tool is constructed, that serves its initial purpose of detecting social distancing violations. With the introduced heatmaps also the functionality is added that it can aid and guide authorities in the process of adjusting spatial planning to prevent the total amount of violations and thereby hopefully the spread of the COVID-19 pandemic. Given the successful tool we were able to generate over the past weeks, we realise that we did not reach our full potential yet and therefore we give some further recommendations at the end of the section. 

Evaluating the requirements set in the introduction we found the following: 

- The execution time of the algorithm on a single frame should be lower than 0.5 sec. → The execution time is clearly limited by the object detection phase, professional hardware has proven to be able to overcome these limitations and will have no problem with the additional steps in the tool.
- The accuracy of people recognition should be higher than 80% → This does not hold for the self-trained network, due to known reasons. The pretrained weights will however satisfy this requirement.
- The accuracy of distance detection should be higher than 80%. → Manual experiments have shown this to be true. A more accurate experiment needs to be performed to be able to make this claim.
- It should be robust in the sense that the program handles a variety of crowds, perspectives, resolutions, lighting conditions and environmental differences. To measure this, 10 clips containing these different conditions are carefully selected as presented section on [Input Data](https://www.notion.so/timovijn/Social-Distance-Monitor-ba55fb2f74b44aed91a729e04bdeb00e#05802bc1d33c4dbaa92b29d6a159a609). → As we carefully selected data based on these features, and the tool works on all of them, we can safely say this is satisfied.
- The operating time should be limited to a 1 time setup with a duration less than 10 minutes. → If the perspective is determined manually, the tool can perform automatically, so this also is a satisfied requirement
- The tool should be user-friendly in the sense that it easy to operate and also outputs a variety of data. → We tried to make the tool as user-friendly as possible, but leave a conclusion on this aspect for the reader to consider.

If the project is split up in two phases: the deep learning phase, were the self trained object detection network was generated and the computer vision phase, where the social distance monitor as a tool itself can be considered we can draw two different sub-conclusions. 

The deep learning phase was less successful in terms of result. Nonetheless we managed to successfully train our own version of the YOLOv3 network on a custom dataset. The results up to this point are not reassuring, but the idea of training the network just for person recognition remains an interesting task for further research.

The computer vision phase was successful in the sense that within a short amount of time accurate results were obtained. From that point on the tool was improved upon several times along the way. 

The process of building this tool has definitely resulted in a great learning curve and we will list our most valuable lessons below. As we are very enthusiastic about the Social Distance Monitor we also state our ideas about the potential and future directions of the Social Distance Monitor.

## Valuable Lessons

- This project has learned us that all data-driven parts of the project come with a lot more complexity than one might initially consider. We have spent countless hours on trying to generate correct images and corresponding labels and trying to shift them to different environments using a variety of formats (zip, tar and uncompressed). Eventually, when the training was initialised, we look back on this phase with struggling feelings, but also realise we have learned a lot. Both of us have the feeling that when similar problems occur in a future occasion, we will be able to troubleshoot and rule out possible causes much quicker. We recommend future students to reserve more time for this process.
- This project has significantly enhanced our programming skills as well as the way one handles a variety of python packages and virtual environments.
- Above all, the steepest learning curve was shown in the field of computer vision, were we went from little experience towards training our own object detection network and applying various computer vision based operations towards video data.
- The social distance monitor has learned us what two students are capable of in a short amount of time, e.g. constructing an application in the field of Artificial Intelligence, that is relevant in the world as we know now and could be deployed largely without needing much modifications. It learned us to plan and execute, from idea generation phase till the documentation of all what we have done.

## Future work

Based upon our findings during the development of the Social Distance Monitor, we would like to discuss some aspects that could be improved in future work. These are also recommendations for people that would like to move forward from this point.

### Couple detection (filtering)

Something that introduces a significant amount of noise in the resulting data is couples (or families) that are walking close to each other or even holding hands. Many restrictive measures do not limit the distance that such persons are allowed to have and they should thus not be identified as a violation.

At this point, the SDM contains no procedure to ignore couples. Such a procedure would probably contain a methodology that ignores all violations between two people that have been in violation for at least an X amount of frames. Methodology more interesting and accurate could even use machine learning to learn to identify couples. This would necessitate labelling video footage, because such a dataset does not exist to our knowledge.

### Automatic perspective

At this point, the SDM has one step that severely limits the applicability of such a tool; the perspective of a scene has to be given manually. Introducing methodology to automatically detect the perspective of a scene would be essential before deploying such a tool on large scale. A limitation that similar social distancing tools that omit the bird’s eye view altogether need not address.

### Redo the training procedure for pedestrian detection only

Despite the fact that our self-trained network does not outperform the pre-trained version of YOLOv3 we still think that a future successful training procedure can increase the accuracy of pedestrian recognition and thereby of the social distance monitor due to the earlier given reasoning. We have reasons to believe this still hold, as our training phase and conditions were far from optimal. 

# References

[1] [YOLOv3: an incremental improvement](https://arxiv.org/pdf/1804.02767.pdf)

[2] [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
