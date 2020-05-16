from imutils.video import VideoStream
import imagezmq


path = "rtsp://192.168.1.77:8080//h264_ulaw.sdp"  # change to your IP stream address
cap = VideoStream(path)

sender = imagezmq.ImageSender(connect_to='tcp://localhost:5566')  # change to IP address and port of server thread
cam_id = 'Camera 2'  # this name will be displayed on the corresponding camera stream

stream = cap.start()

while True:

    frame = stream.read()
    sender.send_image(cam_id, frame)
