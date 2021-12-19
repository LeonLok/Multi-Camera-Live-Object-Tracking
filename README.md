<h1 align='center'>
Multi-Camera Live Object Tracking
</h1>

This repository contains my object detection and tracking projects. All of these can be hosted on a cloud server.

You can also use your own IP cameras with asynchronous processing thanks to [ImageZMQ](https://github.com/jeffbass/imagezmq). I've written a blog post on how to stream using your own smartphones with ImageZMQ [here](https://leonlok.co.uk/blog/live-video-streaming-using-multiple-smartphones-with-imagezmq/).

## Deep SORT and YOLO v4
Check out my [Deep SORT repository](https://github.com/LeonLok/Deep-SORT-YOLOv4) to see the tracking algorithm that I used which includes the options for Tensorflow 2.0, asynchronous video processing, and low confidence track filtering.

***
## Traffic Counting ([Link](https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking/tree/master/traffic_counting))
This project is an extension of the object counting app.

<div align='center'>
<img src="gifs/traffic_counting1.gif" width="80%"/>
</div>

### ([Full video](https://www.youtube.com/watch?v=x6vkXf-mgaw&feature=youtu.be))

### Features

* Trained using a total of **244,617** images generated from the DETRAC dataset. You can find the conversion code that I created [here](https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking/tree/master/detrac_tools). 
    * I used this [paper](https://ieeexplore.ieee.org/document/8909903) as a guideline for data preparation and training.
* Only counts each tracking ID once.
* Counts objects by looking at the intersection of the path of the tracked object and the counting line.
    * Hence, those that lose tracking but are retracked with the same ID still get counted.
* Tracked using low confidence track filtering from the same [paper](https://ieeexplore.ieee.org/document/8909903).
    * Offers much lower false positive rate.
    * Tracked objects show average detection confidence.
    * Tracked classes determined by most common detection class.
* Showing detections is optional (but hides average detection confidence).
* Multiple IP cameras possible.
* Video streaming possible via emulated IP camera.
* Directional counts can be configured based on angle.
* Records counts for every set interval of the hour.
    * Total count.
    * Class-based counts.
* Records intersection details for each counted object.
    * Time of intersection.
    * Coordinate of intersection.
    * Angle of intersection.
* Can be hosted on a cloud server.

Note that since DETRAC doesn't contain any motorcycles, they are the only vehicles that are ignored. Additionally, the DETRAC dataset only contains images of traffic in **China**, so it struggles to correctly detect certain vehicles in other countries due to lack of training data. For example, it can frequently misclassify hatchbacks as SUVs, or not being able to detect taxis due to different colour schemes.


***
## Object Counting ([Link](https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking/tree/master/object_counting))
This project was originally intended to be an app for counting the current number of people in multiple rooms using my own smartphones, where the server would be remotely hosted. Below shows detection, tracking, and counting of people and cars.

<div align='center'>
<img src="gifs/object_counting2.gif" width="50%"/>
</div>

### Features

* Counts the current number of objects in view.
* Tracking is optional.
* Multiple IP cameras possible.
* Records current counts for every set interval of the hour.
    * Current total count.
    * Current class-based counts.
* Can be hosted on a cloud server.

***
## Using my own smartphones as IP cameras

<div align='center'>
<img src="gifs/object_counting1.gif" width="50%"/>
</div>

***
## Training your own vehicle tracking model ([Link](https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking/tree/master/detrac_tools))
I trained a YOLO v4 and Deep SORT model using the [DETRAC](http://detrac-db.rit.albany.edu/) training dataset with v3 annotations. I've provided the scripts for converting the DETRAC training images and v3 annotations into the correct format for training both the YOLO v4 model as well as the Deep SORT tracking model.

I had to train the YOLO v4 model using Darknet and then convert it to Keras format using `convert.py` from the [Keras-to-YOLOv4 repository](https://github.com/Ma-Dan/keras-yolo4). The Deep SORT model was trained using [cosine metric learning](https://github.com/nwojke/cosine_metric_learning).

If you don't want to train your own models, this repository already includes the trained Deep SORT model (mars-small128.pb) from the [original Deep SORT repository](https://github.com/nwojke/cosine_metric_learning). Follow the instructions in the [Keras-to-YOLOv4 repository](https://github.com/Ma-Dan/keras-yolo4) for downloading and converting an already trained Darknet YOLO v4 model to Keras format.Alternatively, you can also find these in my [Deep SORT and YOLOv4 repository](https://github.com/LeonLok/Deep-SORT-YOLOv4)

Please note that if you decide not to train your own models, the vehicle tracking performance will most likely be worse than if you trained your own models on the DETRAC dataset or any other traffic dataset. This is mainly because the original Deep SORT model (mars-small128.pb) was trained on tracking people and not vehicles. However, if your goal is to use this app to count people then this shouldn't be much of an issue.

### Deep SORT conversion parameters
DETRAC images are converted into the Market 1501 training format.

* Occlusion threshold - ignore vehicle sequences with too high occlusion ratio.
* Truncation threshold - ignore vehicle sequences with too high truncation ratio.
* Number of occurrences - vehicle sequences that are too short (i.e. not enough images) are discarded after considering occlusion and truncation ratios.

### YOLO conversion parameters
DETRAC images are converted into the Darknet YOLO training format.

* Occlusion threshold - ignore vehicle sequences with too high occlusion ratio.
* Truncation threshold - ignore vehicle sequences with too high truncation ratio.

Both models were trained and evaluated on the DETRAC training set, but no evaluation has been done yet on the test set due to lack of v3 annotations and I don't have MATLAB for the Deep SORT evaluation software. It's been good enough though for my use case so far. 


***
## Hardware used
* Nvidia GTX 1070 GPU
* i7-8700K CPU

To give some idea of what do expect, I could run two traffic counting streams at around 10fps each (as you can see in the traffic counting gif). Of course, this heavily depends on stream resolution and how many frames are being processed for detection and tracking.

***
## YOLO v3 vs. YOLO v4
I used YOLO v3 when I first started the object counting project which gave me about ~10FPS with tracking, making it difficult to run more than one stream at a time. Using YOLO v4 made it much easier to run two streams with a higher resolution, as well as giving a better detection accuracy.

***
## Dependencies
* Tensorflow-GPU 1.14
* Keras 2.3.1
* opencv-python 4.2.0
* ImageZMQ
* numpy 1.18.2
* Flask 1.1.1
* pillow

This project was built and tested on Python 3.6.
You can use the conda environment file to set up all dependencies.

If environment.yml isn't working, try environment_windows.yml if you're on a Windows machine.

***
## Credits

* https://github.com/miguelgrinberg/flask-video-streaming
* https://github.com/Ma-Dan/keras-yolo4
* https://github.com/nwojke/deep_sort
* https://github.com/Qidian213/deep_sort_yolov3
* https://github.com/yehengchen/Object-Detection-and-Tracking
