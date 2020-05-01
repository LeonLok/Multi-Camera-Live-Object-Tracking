from __future__ import division, print_function, absolute_import

import cv2
from base_camera import BaseCamera

import warnings
import numpy as np
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort.detection import Detection

warnings.filterwarnings('ignore')

class Camera(BaseCamera):
    video_source = 0

    def __init__(self, feed_type, device, port_list):
        super(Camera, self).__init__(feed_type, device, port_list)


    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def yolo_frames(encoder, tracker, device):  # TODO add server name as argument
        yolo = YOLO()
        nms_max_overlap = 1.0

        numFrames = 0

        get_feed_from = ('camera', device)

        while True:
            frame = BaseCamera.get_frame(get_feed_from)
            image_height, image_width = frame.shape[:2]

            if frame is None:
                break

            numFrames += 1

            if numFrames % 2 == 0:

                #image = Image.fromarray(frame)
                image = Image.fromarray(frame[..., ::-1])  # convert bgr to rgb
                boxs = yolo.detect_image(image)[0]
                confidence = yolo.detect_image(image)[1]
                features = encoder(frame, boxs)

                # score to 1.0 here).
                detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in zip(boxs, confidence, features)]

                # Run non-maxima suppression.
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]

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
                    cv2.putText(frame, "Track ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                                2e-3 * frame.shape[0], (0, 255, 0), 1)

                    track_count += 1  # add 1 for each tracked object

                cv2.putText(frame, "Current count: " + str(track_count), (int(20), int(60)), 0, 2e-3 * frame.shape[0],
                            (0, 255, 0), 1)

                for det in detections:
                    bbox = det.to_tlbr()
                    score = "%.2f" % round(det.confidence * 100, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0),
                                  1)  # BLUE BOX
                    cv2.putText(frame, score, (int(bbox[0]), int(bbox[3])), 0,
                                2e-3 * frame.shape[0], (0, 255, 0), 1)

                yield track_count, frame
