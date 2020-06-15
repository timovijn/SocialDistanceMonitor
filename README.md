# Blog post

![Blog%20post%207d35bd1776c6429bb1f55d258b90a908/Screenshot_2020-06-15_at_11.37.02.png](Blog%20post%207d35bd1776c6429bb1f55d258b90a908/Screenshot_2020-06-15_at_11.37.02.png)

---

---

# Introduction

**Problem to solve • Relevant example • Extensibility to other domains • Summary of result • Clear hypothesis and problem statement • Requirements**

With the rise of the COVID-19 pandemic, society calls for innovative solutions to solve the emerging problems and aid humanity in dealing with the new life standards and regulations. One of these regulations is that of social distancing, where civilians are asked to not get any closer to each other than a by government predefined distance. As the goal is to reduce the total amount of COVID-19 contaminations, the need for an application that can show where and how often the regulations are violated is high. This blog post comprises of a tool that evaluates recordings of surveillance cameras and then presents the total amount of violations and the locations where the majority of these violations occur. This way, authorities can adjust the spatial planning accordingly, with the goal to reduce the chance that COVID-19 is spread within their region. For this application to be useful, it should adhere to the following requirements: 

- The runtime of the algorithm on a single frame should be lower than 0.5 sec.
- The accuracy of distance detection should be higher than 90%.
- It should be robust in the sense that the program handles a variety of crowds, perspectives, resolutions, lighting conditions and environmental differences. To measure this, 10 clips containing these different conditions are carefully selected as presented in the method section.
- The operating time should be limited to a 1 time setup with a duration less than 10 minutes.
- The tool should be user-friendly in the sense that it easy to operate and also outputs a variety of data.

This tool was developed during times of the COVID-19 epidemic, however its use is much more versatile than crime prevention and law enforcement only. For example, one can use it to adjust infrastructure or urban planning based on pedestrian flows and measure the effect of events on specific parts of their domain. 

# Related Work

**Other implementations • Underlying principles • YOLO**

For this tool to be completed, inspiration was taken from existing variants and widely used techniques were used to execute specific steps in the program. The other implementations and underlying principles will be set out below: 

### Other Implementations

- Landing AI, a famous start-up in the field of Artificial Intelligence, build a [social distancing detector](https://landing.ai/landing-ai-creates-an-ai-tool-to-help-customers-monitor-social-distancing-in-the-workplace/). The detector works roughly the same as the one presented in this blog post, as they use the video surveillance input, detect pedestrians and evaluate the social distancing violations from a bird's-eye perspective. Where the algorithms differ is in the sense of object detection. Landing AI uses Faster R-CNN, where our program uses YOLOv3, which will be explained later. Landing AI's tool does only show the violations as opposed by our tool, where the locations of occurrence are also given as output.

During the duration of this project, a few others were introduced

### Underlying Principles

Object detection is done by the *[You Only Look Once](https://pjreddie.com/darknet/yolo/)* (YOLO) object detection system. This system requires three files: (1) a weight file; (2) a configuration file; and (3) a name file.

### Libraries

Open CV

# Methods

**Step-by-step walkthrough of algorithm components (visualisation) • Motivation with reference to hypothesis and problem statement**

The *SocialDistancingPedestrian* algorithm consists of a number of constituent parts that will be explained separately in an input-to-output manner.

The overarching goal of the algorithm is to provide the user with insightful indicators indicating the amount and location of social distance violations for an input of pedestrian video footage. On the highest level, the input-to-output relation can thus be visualised as follows.

*Image that shows input video footage and output indicators and in between the “algorithm”.*

When we descend into the lower levels of the algorithm, the constituent parts in the table below can be considered separately. The algorithm progresses through these steps for every frame of a video separately and linearly.

[Untitled](https://www.notion.so/5f97ba769d9b47269c82fdce862f66bb)

## (1) **Video processing**

To obtain (output) a processed video frame from (input) raw video footage, the algorithm contains (1) Video processing.

This step in the algorithm is a straightforward one which starts by loading a file of any video format for the timestamps set by the user, and loading a single frame from this video. Every input video can have its own dimensions. Output is standardised such that frames are always shown in the same horizontal dimension. Furthermore, a frame is always resized to 416 x 416 resolution before continuing to the next step.  Maybe we can implement further video processing such as contrast/colour etc.

Motivated by our requirement of user-friendliness, the algorithm output for this step provides the user with information about the selected video and clip.

```python
...

Started at 12:56:50

...

Checkpoint Initialisation

...

Path: ./Videos/TownCentreXVID.avi
Width: 1920 px
Height: 1080 px
Framerate: 25.0 fps
Duration: 300.0 s
Frames: 7500

...

New frame 0 (0 → 250) (1 of 251) (0%)

...
```

## (2) **Object detection**

To obtain (output) person locations in 3D from (input) a processed video frame, the algorithm contains (2) Object detection.

As stated before, object detection is implemented through the YOLO object detection system. The three required files are stored locally to allow configuration before being loaded for the algorithm to use. The processed frame from one step earlier is further processed by `blob = cv2.dnn.blobFromImage()`. In our case, this function is used to create a 4-dimensional blob from the frame with swapped Blue and Red channels.

The artificial network is initialised by `net = cv2.dnn.readNetFromDarknet()` which takes the weight file and configuration file as input. The input to this network is then set by `net.setInput()` and layer outputs are obtained by `net.forward()`. This output is used to determine bounding boxes of the objects that have been detected to be of class ‘person’ with a confidence value of at least a user set value.

## (3) Perspective change

# Experiments & Data

**What type of data • Data source • How much data • Preprocessing steps • Explicit experiments (Experiment description → Hypothesis → Result) / Performance / Robustness • Ablation study • Shortcomings**

# Conclusion

**Summary • Learning curve • Future work**
