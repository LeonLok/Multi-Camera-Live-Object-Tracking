from base_camera import BaseCamera
import time
import cv2


class Camera(BaseCamera):

    def __init__(self, feed_type, device, image_hub):
        super(Camera, self).__init__(feed_type, device, image_hub)

    @classmethod
    def server_frames(cls, image_hub):
        num_frames = 0
        total_time = 0
        while True:  # main loop
            time_start = time.time()

            cam_id, frame = image_hub.recv_image()
            image_hub.send_reply(b'OK')  # this is needed for the stream to work with REQ/REP pattern

            num_frames += 1

            time_now = time.time()
            total_time += time_now - time_start
            fps = num_frames / total_time

            # uncomment below to see FPS of camera stream
            # cv2.putText(frame, "FPS: %.2f" % fps, (int(20), int(40 * 5e-3 * frame.shape[0])), 0, 2e-3 * frame.shape[0],(255, 255, 255), 2)

            yield cam_id, frame
