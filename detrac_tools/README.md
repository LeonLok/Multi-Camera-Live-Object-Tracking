<h1 align='center'>
Convert DETRAC Dataset for YOLO and Deep SORT
</h1>

## Instructions
First, download and extract the training dataset and v3 training annotations from [DETRAC](http://detrac-db.rit.albany.edu/).

Please note that the conversion process can take a while and requires a decent amount of free disk space. With my default settings on the DETRAC training set, I had **163,171** cropped images of vehicle sequences (which was only about 330MB) and **81,446** copied images for YOLO (which, including annotations, was about 5.13GB).

### Deep SORT (crop_dataset.py)
I followed this [paper](https://ieeexplore.ieee.org/document/8909903) as a guideline for parameters. They removed all sequences of vehicles that had a truncation or occlusion threshold higher than 0.5. Those that then had less than 100 occurrences were ignored. Each vehicle in each satisfactory image is then cropped and resized to 100x100.

I've replicated that in the script so that you can choose your own thresholds and output image size. The script basically goes through the annotations and creates a dictionary of what to crop and what to ignore.

These are the default settings that I used for my own model:
* Occlusion threshold = 0.5
* Trucation threshold = 0.5
* Number of occurrences = 100
* Image size = (100, 100)

You can change these within `crop_dataset.py` or you can pass the arguments in the command line (except for image size).

For example,
```
crop_dataset.py --occlusion_threshold=0.6 --truncation_threshold=0.6 --occurrences=50
```

The cropped sequences should all be outputted to the `DETRAC_cropped` directory in Market 1501 format. Then, follow the instructions on how to train a Market 1501 dataset in the njojke's [cosine metric learning](https://github.com/nwojke/cosine_metric_learning) repository.


### Darknet's YOLO (detrac_to_yolo.py)
I trained a YOLO v4 model using AlexeyAB's [darknet](https://github.com/AlexeyAB/darknet). Follow the instructions there to train your own model.

I decided to include the occlusion and truncation thresholds here for the same reasons as when I trained the Deep SORT model. However, there's no image resize since there's no need to crop anything and no occurrence thresholds since YOLO isn't for tracking.

Like `crop_dataset.py`, you can change the default settings within the file or pass the arguments in the command line.

For example,
```
detrac_to_yolo.py --occlusion_threshold=0.6 --truncation_threshold=0.6
```

The script works in a similar way to `crop_dataset.py` by iterating through each annotation file and calculating the truncation and occlusion ratios of each object. Those that satisfy the requirements are counted with respect to its frame number. Frames that contain no qualifying objects in the end are then ignored.

The images are then copied to the `DETRAC_YOLO_training` directory and the corresponding annotations for each image are also calculated and outputted to `DETRAC_YOLO_annotations`.

A class list text file is also created for the outputted training images under `detrac_classes.txt`. You need this for training in Darknet and also for when you use the YOLO model for object detection.

#### Darknet Config File
I've also included the config file that I used when I was training with Darknet.

