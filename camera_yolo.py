from __future__ import division, print_function, absolute_import

import cv2
from base_camera import BaseCamera

import warnings
import numpy as np
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from importlib import import_module

warnings.filterwarnings('ignore')


class Camera(BaseCamera):
    def __init__(self, feed_type, device, port_list):
        super(Camera, self).__init__(feed_type, device, port_list)

    @staticmethod
    def yolo_frames(unique_name):
        device = unique_name[1]

        tracking = True

        if tracking:
            gdet = import_module('tools.generate_detections')
            nn_matching = import_module('deep_sort.nn_matching')
            Tracker = import_module('deep_sort.tracker').Tracker

            # Definition of the parameters
            max_cosine_distance = 0.3
            nn_budget = None

            # deep_sort
            model_filename = 'model_data/mars-small128.pb'
            encoder = gdet.create_box_encoder(model_filename, batch_size=1)

            metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
            tracker = Tracker(metric)

        yolo = YOLO()
        nms_max_overlap = 1.0

        num_frames = 0

        get_feed_from = ('camera', device)

        while True:
            frame = BaseCamera.get_frame(get_feed_from)
            # image_height, image_width = frame.shape[:2]

            if frame is None:
                break

            num_frames += 1

            if num_frames % 2 != 0:  # only process frames at set number of frame intervals
                continue

            #image = Image.fromarray(frame)
            image = Image.fromarray(frame[..., ::-1])  # convert bgr to rgb
            boxes, confidence, classes = yolo.detect_image(image)
            if tracking:
                features = encoder(frame, boxes)

                detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in zip(boxes, confidence, features)]
            else:
                detections = [Detection_YOLO(bbox, confidence) for bbox, confidence in zip(boxes, confidence)]
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            if tracking:
                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                track_count = int(0)  # reset counter to 0

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255),
                                  1)  # WHITE BOX
                    cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                                1.5e-3 * frame.shape[0], (0, 255, 0), 1)

                    track_count += 1  # add 1 for each tracked object

                cv2.putText(frame, "Current count: " + str(track_count), (int(20), int(40)), 0, 2e-3 * frame.shape[0],
                            (255, 255, 255), 2)

            det_count = int(0)
            for det in detections:
                bbox = det.to_tlbr()
                score = "%.2f" % (det.confidence * 100) + "%"
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0),
                              1)  # BLUE BOX
                #if len(classes) > 0:
                cls = classes[0][0]
                cv2.putText(frame, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                            1.5e-3 * frame.shape[0], (0, 255, 0), 1)

                det_count += 1

            if tracking:
                count = track_count
            else:
                count = det_count

            yield count, frame
