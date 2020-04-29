import time
import threading
from importlib import import_module
import imagezmq
import datetime
import os

try:
    from greenlet import getcurrent as get_ident
except ImportError:
    try:
        from thread import get_ident
    except ImportError:
        from _thread import get_ident


class CameraEvent:
    """An Event-like class that signals all active clients when a new frame is
    available.
    """
    def __init__(self):
        self.events = {}

    def wait(self):
        """Invoked from each client's thread to wait for the next frame."""
        ident = get_ident()
        if ident not in self.events:
            # this is a new client
            # add an entry for it in the self.events dict
            # each entry has two elements, a threading.Event() and a timestamp
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self):
        """Invoked by the camera thread when a new frame is available."""
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                # if this client's event is not set, then set it
                # also update the last set timestamp to now
                event[0].set()
                event[1] = now
            else:
                # if the client's event is already set, it means the client
                # did not process a previous frame
                # if the event stays set for more than 5 seconds, then assume
                # the client is gone and remove it
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self):
        """Invoked from each client's thread after a frame was processed."""
        self.events[get_ident()][0].clear()


class BaseCamera:
    threads = {}  # background thread that reads frames from camera
    frame = {}  # current frame is stored here by background thread
    last_access = {}  # time of last client access to the camera
    event = {}

    def __init__(self, feed_type, device, port_list):
        """Start the background camera thread if it isn't running yet."""
        self.unique_name = (feed_type, device)
        BaseCamera.event[self.unique_name] = CameraEvent()

        if self.unique_name not in BaseCamera.threads:
            BaseCamera.threads[self.unique_name] = None
        if BaseCamera.threads[self.unique_name] is None:
            BaseCamera.last_access[self.unique_name] = time.time()

            # start background frame thread
            BaseCamera.threads[self.unique_name] = threading.Thread(target=self._thread,
                                                                    args=(self.unique_name, port_list))
            BaseCamera.threads[self.unique_name].start()

            # wait until frames are available
            while self.get_frame(self.unique_name) is None:
                time.sleep(0)

    @classmethod
    def get_frame(cls, unique_name):
        """Return the current camera frame."""
        BaseCamera.last_access[unique_name] = time.time()

        # wait for a signal from the camera thread
        BaseCamera.event[unique_name].wait()
        BaseCamera.event[unique_name].clear()

        return BaseCamera.frame[unique_name]

    @staticmethod
    def frames():
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses')

    @staticmethod
    def yolo_frames(source, encoder, tracker):
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses')

    @staticmethod
    def server_frames(image_hub):
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses')

    @classmethod
    def yolo_thread(cls, unique_name):
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

        device = unique_name[1]
        frames_iterator = cls.yolo_frames(encoder, tracker, device)

        current_date = datetime.datetime.now().date()
        count_dict = {}  # initiate dict for storing counts
        try:
            for track_count, frame in frames_iterator:
                now = datetime.datetime.now()
                rounded_now = now - datetime.timedelta(microseconds=now.microsecond)  # round to nearest second
                current_minute = now.time().minute
                if current_minute == 0 and len(count_dict) > 1:
                    count_dict = {}  # reset counts every hour
                else:
                    write_interval = 10
                    if current_minute % write_interval == 0:  # write to file once only every write_interval minutes
                        if current_minute not in count_dict:
                            count_dict[current_minute] = True
                            print('Writing current count ({}) to file.'.format(track_count))
                            output_filename = 'counts for {}, camera {}.txt'.format(current_date, device)
                            count_file = open(output_filename, 'a')
                            count_file.write(str(rounded_now) + ", " + device + ', ' + str(track_count) + "\n")
                            count_file.close()
                BaseCamera.frame[unique_name] = frame
                BaseCamera.event[unique_name].set()  # send signal to clients
                time.sleep(0)
                if time.time() - BaseCamera.last_access[unique_name] > 60:
                    frames_iterator.close()
                    print('Stopping YOLO thread due to inactivity')
                    pass
        except Exception:
            BaseCamera.event[unique_name].set()  # send signal to clients
            frames_iterator.close()
            print('Stopping YOLO thread ({}) due to error.'.format(device))

    @classmethod
    def server_thread(cls, unique_name, port):
        device = unique_name[1]
        print(device, port)

        image_hub = imagezmq.ImageHub(open_port='tcp://*:{}'.format(port))
        frames_iterator = cls.server_frames(image_hub)

        try:
            for frame in frames_iterator:
                BaseCamera.frame[unique_name] = frame
                BaseCamera.event[unique_name].set()  # send signal to clients
                time.sleep(0)
                if time.time() - BaseCamera.last_access[unique_name] > 5:
                    frames_iterator.close()
                    image_hub.zmq_socket.close()
                    print('Closing server socket.')
                    print('Stopping camera thread ({}) due to inactivity'.format(device))
                    pass
        except Exception:
            frames_iterator.close()
            image_hub.zmq_socket.close()

            print('Closing server socket.')
            print('Stopping camera thread  due to error.')

    @classmethod
    def _thread(cls, unique_name, port_list):  # TODO add kwargs for source for YOLO threads?
        feed_type = unique_name[0]
        device = unique_name[1]
        if feed_type == 'camera':
            print('Starting server thread ({})'.format(unique_name))
            port = port_list[int(device)]
            cls.server_thread(unique_name, port)

        elif feed_type == 'yolo':
            """Camera background thread."""
            print('Starting YOLO thread ({})'.format(device))
            cls.yolo_thread(unique_name)

        BaseCamera.threads[unique_name] = None
