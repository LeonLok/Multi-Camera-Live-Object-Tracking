from base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0

    def __init__(self, feed_type, device, image_hub):
        super(Camera, self).__init__(feed_type, device, image_hub)

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @classmethod
    def server_frames(cls, image_hub):

        while True:  # main loop

            cam_id, frame = image_hub.recv_image()
            image_hub.send_reply(b'OK')  # this is needed for the stream to work with REQ/REP pattern

            yield frame
