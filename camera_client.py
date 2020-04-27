from imutils.video import VideoStream
import imagezmq


path = "rtsp://192.168.1.70:8080//h264_ulaw.sdp"
cap = VideoStream(path)

sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555')
cam_id = '0'

stream = cap.start()

while True:

    frame = stream.read()
    sender.send_image(cam_id, frame)
