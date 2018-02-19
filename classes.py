import cv2
import numpy as np
import threading

class reader:

    def __init__(self, filename):
        self.video_capture = cv2.VideoCapture(filename)
        self.width = int(self.video_capture.get(3))
        self.height = int(self.video_capture.get(4))
        self.execute = False

        #Prepare frame buffer
        _, self.frame = self.video_capture.read()
        self.frame = self.frame.astype(np.float32)
        self.frame = np.divide(self.frame, 255.0)

    def read(self):
      while True:
        while not self.execute: continue #wait for main thread
        self.currentframe = self.frame
        _, self.frame = self.video_capture.read()
        if self.frame is None:
            exit("End of video file!")

        self.frame = self.frame.astype(np.float32) #Change type to be compatible with edgeDetect
        self.frame = np.divide(self.frame, 255.0) #Normalize to [0,1]
        self.execute = False

        print(self.currentframe[100][100])

class generator:

    def __init__(self, modelfile, video):

        self.Reader = video #multimedia reader

        print("Loading model0...")
        self.edgeGenerator0 = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)
        print("Loading model1...")
        self.edgeGenerator1 = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)

        self.run0 = True #flags controlling generate threads
        self.run1 = True

        Thread0 = threading.Thread(target=self._generate0) #Thread0 handles even frames
        Thread1 = threading.Thread(target=self._generate1) #Thread1 handles odd frames
        Thread0.daemon = True
        Thread1.daemon = True
        Thread0.start()
        print("initializing map0...")
        while self.run0: continue #wait for first thread to create map
        Thread1.start()
        print("initializing map1...")
        while self.run1: continue #wait for second thread to create map

        self.execute = True #flags controlling this thread
        self.frame = 0

    def generate(self):
      while True: #continuously execute
        while not self.execute: continue

        if self.frame % 2 == 0: #even
            self.current_edgearray = self.suppressed_edgearray0.copy()
            self.current_orientationarray = self.orientationarray0.copy()
            self.run0 = True
            self.frame = self.frame + 1
        elif self.frame % 2 != 0: #odd
            self.current_edgearray = self.suppressed_edgearray1.copy()
            self.current_orientationarray = self.orientationarray1.copy()
            self.run1 = True
            self.frame = self.frame + 1

        self.execute = False

    def _generate0(self):
      while True:
        while not self.run0: continue
        edgearray0 = self.edgeGenerator0.detectEdges(self.Reader.currentframe)
        self.Reader.execute = True
        self.orientationarray0 = self.edgeGenerator0.computeOrientation(edgearray0)
        self.suppressed_edgearray0 = self.edgeGenerator0.edgesNms(edgearray0, self.orientationarray0)
        self.run0 = False

    def _generate1(self):
      while True:
        while not self.run1: continue
        edgearray1 = self.edgeGenerator1.detectEdges(self.Reader.currentframe)
        self.Reader.execute = True
        self.orientationarray1 = self.edgeGenerator1.computeOrientation(edgearray1)
        self.suppressed_edgearray1 = self.edgeGenerator1.edgesNms(edgearray1, self.orientationarray1)
        self.run1 = False
