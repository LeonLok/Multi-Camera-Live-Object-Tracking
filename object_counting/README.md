<h1 align='center'>
Flask Multi-Camera Object Counting
</h1>

A Flask app for multiple live video streaming over a network with object detection, tracking (optional), and counting. Uses YOLO v4 with Tensorflow backend as the object detection model and Deep SORT trained on the MARS dataset for object tracking. Each video stream has an independent thread and uses ImageZMQ for asynchronous sending and processing of frames.

Much of the way this app works is based on Miguel Grinberg's [Flask video streaming](https://github.com/miguelgrinberg/flask-video-streaming). Since this is a Flask app, the video streams are accessed via web browser and it's possible to have more than one web client. If all the web clients disconnect from the app, then every video stream thread will automatically shutdown due to inactivity after a set time period. The video streams will restart once a web client connects again, but unlike Miguel's app, the camera clients that are sending the frames must be restarted.


***
## Camera client set-up
Enter the camera stream addresses for your own cameras in each client script and make sure that each camera client is sending frames to the correct server address and port. 

The video_feed function in app.py contains a list of the assigned ports for each server called `port_list`. Currently, camera_client_0.py is device number 0 and it uses the first port in the list (5555). camera_client_1.py is device number 1 and it uses the second port in the list (5566). Basically, the device number corresponds to the index in `port_list`, so they need to match accordingly. If you want to add another camera, create a camera_client_2.py with a new port and add  this port to `port_list` in app.py as the third port.

You also need to make sure that the templates/index.html file contains the correct number of streams for however many you want to be displayed.

For example, if you want to activate the second camera and YOLO stream, make sure that these are uncommented:
```
<img src="{{ url_for('video_feed', feed_type='camera', device=1) }}"...
```
and 
```
<img src="{{ url_for('video_feed', feed_type='yolo', device=1) }}"...
```

You can add more streams by following the same pattern using device=2 and so on.

If you want to learn more about how the server and clients work with ZeroMQ, see ![ImageZMQ](https://github.com/jeffbass/imagezmq). The current default messaging pattern used is the request-reply pattern.

### Changing camera stream names
To change the name of each camera stream shown in the top-left corner, edit `cam_id` in the camera client files.

***
## Model
This app uses YOLO v4 weights that are converted from Darknet to Keras format. You need to train or convert your own and put it in the model_data folder. See this [repository](https://github.com/Ma-Dan/keras-yolo4).

To modify the detection settings like IOU threshold, anchors, class names etc., you can do so in yolo.py.

### Detecting multiple classes
Modify line 103 in yolo.py. For example, to detect people and cars, change it to:
```
            if predicted_class != 'person' and predicted_class != 'car':
                continue
```

***
## Deep SORT
Deep SORT is used for object tracking. However, Please note that the tracking model is only trained for tracking people, so you'd need to train a model yourself for tracking other objects. As you can see in the demonstration gifs, it can still work for tracking other objects like cars. 

See [cosine metric learning](https://github.com/nwojke/cosine_metric_learning) to train your own Deep SORT tracking model.

### Disabling Deep SORT
However, if you don't want to use tracking at all, you can turn it off by changing line 28 from 
```
tracking = True
```
to
```
tracking = False
```

***
## Running locally
First, configure each camera client so that they have the correct camera stream address. Also make sure that the frames are being sent to localhost with the correct port (e.g. tcp://localhost:5555). As mentioned earlier, you can also change the name of the camera stream by changing `cam_id` to something more relevant.

Before running, check that templates/index.html is correctly configured.

Run app.py and then start running each camera client. Once everything is running, launch a web browser and enter localhost:5000. That should open index.html in the browser and the video threads should start running. The camera streams should load pretty quickly; if they don't, then try restarting the camera clients and refresh the browser. The YOLO streams will also load eventually, but they take longer to load due to starting Tensorflow and loading the YOLO model.

If the YOLO thread is shutting down before it finishes loading, you need to increase the time limit on line 117:
```
if time.time() - BaseCamera.last_access[unique_name] > 60:
```
All threads will shutdown after this time limit if the app thinks there's no more viewing web clients. If the YOLO streams are not used, then the default time limit for camera stream threads to shutdown is 5 seconds.

***
## Running remotely
The process is similar to running locally. If you have your own remote server, configure each camera client so that they'll be sending frames to this server address with the correct port instead of localhost (e.g. tcp://server-address-here:5555). 

Clone the repository on the remote server and check that it has the correct ports forwarded so that your browser and camera clients can connect to it. Run app.py and then start running the camera clients. Like before, you should now be able to connect to the app by entering the server address with port 5000 into the browser (i.e. replace localhost:5000 with server-address-here:5000). Again, if nothing loads, try restarting the camera clients and refresh the browser.

***
## Counts
The total current object counts are automatically stored in a text file every set interval of the hour for every detected object. Each newly detected class also creates a new class counts file to store the current counts for that class, and will also appear as text in the YOLO stream. 

***
## Performance
Hardware used:
* Nvidia GTX 1070 GPU
* i7-8700K CPU

Hosting the  on a local server gave ~15FPS on average with a single camera stream at 640x480 resolution streamed locally at 30FPS. Turning off tracking gave me ~16FPS. As you'd expect, having multiple streams will lower the FPS significantly as shown in the demonstration gifs.

There's a lot of other factors that can impact performance like network speed and bandwidth, but hopefully that gives you some idea on how it'll perform on your machine.

Lowering the resolution or quality of the stream will improve performance but also lower the detection accuracy. 

