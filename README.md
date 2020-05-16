# Flask Multi-Camera Streaming With YOLO v4 and Deep SORT
Multiple live video streaming over a network with object detection, tracking (optional), and counting. Uses YOLO v4 with Tensorflow backend as the object detection model and Deep SORT trained on the MARS dataset for object tracking. Each video stream uses ImageZMQ for asynchronous processing.

See my other repository for only YOLO v4 and Deep SORT:
https://github.com/LeonLok/Deep-SORT-YOLOv4

## Model
This application uses YOLO v4 weights that were converted from Darknet to Keras format. To train or convert your own, see https://github.com/Ma-Dan/keras-yolo4.

## Demonstration
The main goal is to be able to run this on an cloud server with multiple camera streams.

First, run app.py and then start running each camera client. Once everything is running, launch a browser and enter the server address. Below is a quick demonstration of the application hosted on a local server (localhost:5000):

![](demonstration.gif)

To test this project yourself, make sure that each camera client is sending frames to the correct address and port. The video_feed function in app.py contains a list of the assigned ports for each server called port_list. Device 0 corresponds to the first port, device 1 corresponds to the second port, etc. The templates/index.html file also needs to be changed depending on how many camera streams you want to be displayed.

Currently, camera_client_0.py is device 0 and so it currently uses the first port (5555). camera_client_1.py is device 1 and so it currently uses the second port (5566). If you want to add more cameras, create a camera_client_2.py and add a third port to the list of ports in app.py. Then update templates/index.html to include the third camera stream and YOLO stream.

If you want to learn more about how the server and clients work, see ![ImageZMQ](https://github.com/jeffbass/imagezmq).

## Detecting Multiple Classes
Modify line 103 in yolo.py. For example, to detect people and cars, change it to:
```
            if predicted_class != 'person' and predicted_class != 'car':
                continue
```

## Disabling Deep SORT
Change line 28 from 
```
tracking = True
```
to
```
tracking = False
```

## Counts
The total current object count is automatically stored in a text file every set interval of the hour for every detected object. Each newly detected class also creates a new class counts file to store the current count for that class and will also appear in the YOLO stream. 

## Performance
Hardware used:
* Nvidia GTX 1070 GPU
* i7-8700K CPU

Hosting the application on a local server gave ~15FPS on average with a single camera stream at 640x480 resolution streamed locally at 30FPS. Turning off tracking gave me ~16FPS. As you'd expect, having multiple streams will lower the FPS significantly as shown in the demonstration gif above.

There's a lot of other factors that can impact performance like network speed and bandwidth, but hopefully that gives you some idea on how it'll perform on your machine.

Lowering the resolution or quality of the stream will improve performance but also lower the detection accuracy. 

### YOLO v3 vs. YOLO v4
I used YOLO v3 when I first started this project which gave me about ~10FPS with tracking, making it difficult to run more than one stream at a time. Using YOLO v4 made it much easier to run two streams with a higher resolution, as well as giving a better detection accuracy.

## Dependencies
* Tensorflow-GPU 1.14
* Keras 2.3.1
* opencv-python 4.2.0
* ImageZMQ
* numpy 1.18.2
* Flask 1.1.1
* pillow

This project was built and tested on Python 3.6.

### Credits
This project was built with the help of:
* https://github.com/miguelgrinberg/flask-video-streaming
  * https://github.com/miguelgrinberg/flask-video-streaming/issues/11#issuecomment-343605510
* https://github.com/Ma-Dan/keras-yolo4
* https://github.com/Qidian213/deep_sort_yolov3
* https://github.com/yehengchen/Object-Detection-and-Tracking
