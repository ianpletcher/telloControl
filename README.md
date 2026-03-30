# Tello Drone Control with YOLOv8 Inference Model

telloControl is a companion project to [droneControl](https://github.com/ianpletcher/droneControl) that integrates droneControl's main detection and tracking logic with the *djitellopy* Python library. The drone's operating procedure involves receiving video, running and tracking detections, rendering graphics, and sending commands based on video stream from a DJI Tello drone.

## High-Level Architecture:

Camera &rarr; YOLOv8 inference &rarr; Centroid Tracker &rarr; OpenCV overlay graphics rendering

### Command Architecture:

OpenCV Mouse Callback &rarr; Drone Control Loop &rarr; Velocity Command Computation &rarr; Drone Control Loop (tello library calls)

## Rendering/Tracking Components:

### Camera:

The camera stream is established with the built-in *djitellopy* library functions.

```python
tello.streamon()
...
frame_read = tello.get_frame_read()
```
tello.streamon() initializes the camera stream, and tello.get_frame_read() fetches self.background_frame_read() from the Tello class.

We work with individual frames by accessing *frame_read.frame* and scaling this to the Tello's default resolution.

### YOLOv8 Inference

The YOLOv8 model is accessed using the *ultralytics* Python library. The *yolov8s.pt* model is fetched, then the model is run on the frame, which is passed as a scaled numpy array according to the global frame dimensions. Each bounding box is processed to find pixel coordinates, confidence score, and class label. A 5x5 pixel array is sampled to find the average color around the centroid. Each detection in the frame initializes a dictionary containing its bounding box coordinates, centroid coordinates, class label, confidence score, and color. Each of these detection dicts is appended to a list.

```python
# in main.py
from ultralytics import YOLO # ultralytics library for working with YOLO model
...
YOLO_MODEL = 'yolov8s.pt' # YOLOv8 model
...
model = YOLO(YOLO_MODEL)
...
detections = run_yolo_inference(model, frame, app_state) # Returns list of detection dictionaries {'bbox', 'centroid', 'label', 'confidence', 'color',}
```

### Centroid Tracker
The app's tracker object is updated on each frame using one of two methods. If a target has been selected, only the target's detection info is updated in the global tracked objects dictionary. Otherwise, all detections are updated and everything in the tracked objects dictionary is constantly modified. Updating only the target when selected aims to save unnecessary computational overhead due to tracking irrelevant objects.

Primary methods:
- *update_all_detections*
- *update_target*

#### update_all_detections
All detections in frame (filtered by confidence score) are updated. First, the detections are matched to previously confirmed tracks (detections we know are valid and persistent). This pass matches current detection's positions against velocity-predicted positions, and uses color matching as a fallback. Each detection that goes unmatched is tested for high overlap with a confirmed track to avoid registering new tentative tracks for the same object*. Then, in a second pass, The detections that are unmatched by the first step are matched against candidates for confirmed tracks. If a match is made, the tentative track is promoted to a confirmed track to be matched against in the first pass on the next frame. Each detection that goes unmatched by both passes becomes a tentative track.

### update_target
Lightweight version of *update_all_detections* that matches against only the target's track. Velocity-based position predictions and color matching are still used to search for a match among the current in-frame detections. Since this object is the target (confirmed by user), matching only occurs in this single pass.

*This may need fine-tuning to avoid prematurely filtering out adjacent objects. This also may not reliably catch duplicates as they may be mistaken for frame updates (i.e. where the object has moved in the frame) during the first pass of filetering.

### OpenCV Graphics Rendering

Bounding boxes with IDs and confidence, as well as an application state head-up display are rendered to the video stream. Each object in the tracked items ordered dict has its bounding box drawn to the frame using the OpenCV rectangle function. The head-up display shows the drone's state, current target ID, battery percentage, and the permissable commands to operate the drone.

## Drone Command/Control Components

### OpenCV Mouse Callback

The *make_mouse_callback* function waits for a left-click event. If the click is within the dimensions of a bounding box, the target id is updated to that box's object id. The drone's state is then set to "TRACKING". If an empty space is clicked, the target is cleared.

### Drone Control Loop

The control loop waits for an app shutdown event to occur. The drone's operation can exist in one of the four following states:

["MANUAL], ["TRACKING"], ["HOVERING"], ["RETURNING"]

The control loop sets states and executes commands based on the following logic:

If the drone's state is set to "MANUAL":
- If the drone is in the air, a blank rc control message is sent to the Tello (0, 0, 0, 0). 

If the drone's state is set to "TRACKING": 
- If there is no target selected, the target is cleared and the state is returned to "MANUAL". 
- If a target has been selected but lacks detection data (i.e. target has been lost), the drone's state is set to "HOVERING".
    - If the drone is airborne, a blank rc control message is sent to the Tello (0, 0, 0, 0)
- If a target has been selected and there is detection data, the *compute_velocity_command* function is called.
    - If the drone is airborne, the resulting rc control messsage is sent to the Tello.

If the drone's state is set to "HOVERING":
The elapsed time since the target was lost is tracked.
The maximum allowed hover timeout is compared with elapsed time.
- If target data repopulates, the drone is returned to "TRACKING".
- If elapsed time surpasses the amount permitted by HOVER_TIMEOUT, the drone's state is set to "RETURNING".
    - If the drone is airborne, a blank rc control message is sent to the Tello (0, 0, 0, 0).
- If elapsed time is still within the HOVER_TIMEOUT threshold:
    - If the drone is airborne, a blank rc control message is sent to the drone until either:
        - Target is reacquired *OR*
        - Elapsed time surpasses HOVER_TIMEOUT threshold.

If the drone's state is set to "RETURNING":
The elapsed time since the HOVER_TIMEOUT threshold was surpassed is tracked.
- If elapsed time surpasses the RETURN_HOLD_DURATION threshold, the drone's state is set to "MANUAL".

There is only one case where nonempty commmands are sent to the Tello: If the drone is in "TRACKING" mode, a target is selected, and there is detection data in the *tracked* Ordered Dict corresponding to the target. Otherwise, no commands or blank rc controls are relayed to the Tello.

### Velocity Command Computation

The *compute_velocity_commands* function takes in *target_data*, *frame_width*, and *frame_height*. The target bounding box centroid and frame centroid are computed, and the target area is computed as a fraction of the frame size.

The distance between the frame centroid and the target centroid is used in dimensional components to determine values for axis-designated commands. Directional distances have the following correspondences:

- YAW : Horizontal distance between frame centroid and target centroid (LEFT-RIGHT)
- VERT: Vertical distance between frame centroid and target centroid (UP-DOWN)
- FWD: Difference in target bbox area and frame area (FORWARD)

Each of these values are clamped using *np.clip* to predefined maximum and minimum. This is to avoid cases where, due to faulty detection or other errors, a command parameter that is too extreme is communicated to the Tello. Additionally, the *YAW* and *VERT* components are tested against the DEADBAND_PX threshold to ensure that small distances between frames do not cause erratic motion.

These values are used to send an rc control message to the Tello in the format:
(0, fwd, vert, yaw)

*Note that we default to yaw rather than left-right to avoid potential mismatches or redundant computation.

## Getting started:

### Dependencies:

Running this project requires the following dependencies (whole or components):

- *djitellopy*
- *threading*
- *OpenCV*
- *time*
- *sys*
- *ultralytics*
- *collections*
- *logging*
- *numpy*
- *scipy.spatial.distance*

### Running the software:

To run the software, first power on the Tello drone, ensuring that there is sufficient battery life. Then, enter your PC's network settings and connect to the Tello's wifi broadcast (name like *"TELLOXXXX"*). Then, run the code in the python environment where the dependencies have been installed by entering: 

```bash
python3 main.py
```

to the terminal.

Bounding boxes will appear around confirmed detections, where a detected object can be designated as target by clicking within its bounding box. The following keyboard commands can be used for control:

- T: Takeoff
- L: Land
- Q/Esc: Quit

### Video Player:

Each run of *main.py* saves videos to the *videos* list in *review_footage.py*. You can run:

```bash
python3 review_footage.py
```

To review video data taken from the Tello*.

*Testing needed.