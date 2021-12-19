<h1 align='center'>
Flask Multi-Camera Traffic Counting
</h1>

This project was built from the [object counting app](https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking/tree/master/object_counting) so the instructions are the same. The main difference is that I removed the normal camera streams in templates/index.html, leaving only the YOLO streams to be activated.

If you're getting the CUDNN_STATUS_ALLOC_FAILED error, try removing the commented lines at the top of yolo.py.

***
## Vehicle detection and tracking models
I trained a YOLO v4 model using Darknet and then converted it to Keras format using `convert.py` from the [Keras-to-YOLOv4 repository](https://github.com/Ma-Dan/keras-yolo4). I also trained a Deep SORT model using [cosine metric learning](https://github.com/nwojke/cosine_metric_learning). 

If you don't want to train your own YOLO or Deep SORT models, then you can go to my [Deep SORT and YOLOv4 repository](https://github.com/LeonLok/Deep-SORT-YOLOv4) and use the included Deep SORT model from [cosine metric learning](https://github.com/nwojke/cosine_metric_learning). It also includes the previously mentioned `convert.py` script. Follow the instructions in the [Keras-to-YOLOv4 repository](https://github.com/Ma-Dan/keras-yolo4) for downloading and converting an already trained Darknet YOLO v4 model to Keras format. 

I've created scripts for converting the DETRAC dataset into the correct format for training [here](https://github.com/LeonLok/Multi-Camera-Live-Object-Tracking/tree/master/detrac_tools). After training and converting the weights, put the weights and classes list in the `model_data` folder.

The YOLO model, anchors, and classes paths need to be changed accordingly in the `yolo.py` file, and you can also modify the IOU thresholds here.

The Deep SORT model path on line 56 in `camera_yolo.py` also needs to be changed.

### Tracking specific classes
I've commented out the following part (line 103 and 104) in `yolo.py`:
```
            if predicted_class != 'person':
                continue
```
This is because I'm using the DETRAC class list, which are all vehicle classes.

If you want to track only specific types of vehicles, then modify this part of the code accordingly. Also, if you want to group certain classes together, then you can also modify the class list by renaming them to the same class name.

***
## Average detection confidence
To change the average detection confidence threshold, go to `deep_sort/tracker.py` and modify line 40. You can also change the minimum number of steps required for confirming tracks here. 

***
## Counting line
The app counts tracked objects by seeing if the line drawn from the current tracked position to the most recently tracked position intersects the counting line. Hence, objects will be counted as long as they maintain the same tracking ID once they cross the line. 

The counting line can be adjusted on line 112 in `camera_yolo.py`:

```
            if cam_id == 'Camera 1':
                line = [(0, int(0.5 * frame.shape[0])), (int(frame.shape[1]), int(0.5 * frame.shape[0]))]
            else:
                line = [(0, int(0.33 * frame.shape[0])), (int(frame.shape[1]), int(0.33 * frame.shape[0]))]
```

Each stream can have a different line in this way. 

### Lines with different angles
(If directional counts are not important then this section can be skipped.)

It's recommended to change line 153 in `camera_yolo.py` if you want to change the angle of the line (e.g. a vertical or diagonal line):

```
                    if angle > 0:
                        up_count += 1
                    if angle < 0:
                        down_count += 1
```

The angle of the tracked object upon intersection with the line is calculated with respect to the positive x-axis. If the angle is within +180 degrees then it's counted as upwards, and if the angle is within -180 degrees then it's counted as downwards. 

With a vertical line for example, you would need to add left and right counts where angles that are greater than 90 or less than -90 degrees would be counted as left, and angles that are between 90 and -90 degrees are counted as right.

***
## Showing detections

By default, the average detection confidence and the **most common** class will be displayed for each tracked object. To show the latest detection class and corresponding detection confidence instead, simply change line 45 in `camera_yolo.py` from `show_detections = False` to `show_detections = True`.



***
## Count data
All data is stored in CSV format and are written to the `counts` directory every set interval of the hour. Within here will be subdirectories which separates the data by date, and within each date subdirectory will contain a directory for:
* total counts,
* class-based counts,
* and intersections.

The intersection file contains information about time, coordinates, and angle of each intersection.

To change the write interval, go to line 206 in `camera_yolo.py`. It's set to 5 by default, so it'll write counts every 5 minutes of the hour, e.g. 17:00, 17:05, 17:10... etc. regardless of when the app was started.
