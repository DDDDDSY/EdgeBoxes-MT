import cv2
import numpy as np

class reader:

    def __init__(self, filename):
        self.video_capture = cv2.VideoCapture(filename)
        self.width = int(self.video_capture.get(3))
        self.height = int(self.video_capture.get(4))
        self.execute = False

        #Prepare frame buffer
        _, self.frame = self.video_capture.read()
        self.frame = self.frame.astype(np.float32)
        self.frame = np.apply_along_axis(self.normalize, 0, self.frame)

    def normalize(self, a): #Normalize to a float in [0,1]
        return np.float32(a/255)

    def read(self):
        while not self.execute: continue #wait for main thread
        self.currentframe = self.frame
        _, self.frame = self.video_capture.read()
        if self.frame is None:
            exit("End of video file!")

        self.frame = self.frame.astype(np.float32) #Change type to be compatible with edgeDetect
        self.frame = np.apply_along_axis(self.normalize, 0, self.frame) #Apply normalize function over array
        self.execute = False
