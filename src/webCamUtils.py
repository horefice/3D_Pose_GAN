# import the necessary packages
from threading import Thread
import cv2 as cv

class WebcamVideoStream:
    def __init__(self, src=0, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.name = name
        self.oneshot = False
        print(src)
        if str(src).lower().endswith(('.png', '.jpg')):
            self.frame = cv.imread(src)
            self.stopped = False
            self.oneshot = True
        else:
            self.oneshot = False
            self.stream = cv.VideoCapture(src)
            (self.grabbed, self.frame) = self.stream.read()
            self.stopped = False
            self.start()

    def isOpened(self):
        return not self.stopped

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        if self.oneshot == True:
            self.stop()

        if self.frame is not None:
            return True, self.frame
        else:
            print("return None frame")
            return False, None

    def release(self):
        self.stop()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True