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

        self.framenum = 0

    def read(self):
      while True:
        while not self.execute: continue #wait for main thread
        self.currentframe = self.frame
        self.framenum = self.framenum + 1
        _, self.frame = self.video_capture.read()
        if self.frame is None:
            exit("End of video file!")

        self.frame = self.frame.astype(np.float32) #Change type to be compatible with edgeDetect
        self.frame = np.divide(self.frame, 255.0) #Normalize to [0,1]
        self.execute = False

class generator:

    def __init__(self, modelfile, video):

        self.Reader = video #multimedia reader

        print("Loading model0...")
        self.edgeGenerator0 = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)
        print("Loading model1...")
        #self.edgeGenerator1 = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)
        print("Loading model2...")
        #self.edgeGenerator2 = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)

        self.frame = 0 #Frame count

        self.run0 = True #flags controlling generate threads
        self.run1 = True
        self.ready0 = False #flags indicating state of generate threads' data
        self.ready1 = False

        self.run2 = True #third thread
        self.ready2 = False

        Thread0 = threading.Thread(target=self._generate0)
        Thread0.daemon = True
        Thread0.start()
        print("initializing map0...")
        while not self.ready0: continue #wait for first thread to create map

        Thread1 = threading.Thread(target=self._generate1)
        Thread1.daemon = True
        #Thread1.start()
        print("initializing map1...")
        #while not self.ready1: continue #wait for second thread to create map

        Thread2 = threading.Thread(target=self._generate2)
        Thread2.daemon = True
        #Thread2.start()
        print("initializing map2...")
        #while not self.ready2: continue #wait for third thread to create map

        self.execute = False #flags controlling this thread
        self.threadnum = 0

    def generate(self):
      while True: #continuously execute
        while not self.execute: continue

        if self.threadnum == 0: #Thread0
            while not self.ready0: continue
            self.ready0 = False
            self.current_edgearray = self.suppressed_edgearray0.copy()
            self.current_orientationarray = self.orientationarray0.copy()
            self.run0 = True
            self.frame = self.frame + 1
            self.threadnum = 0

        elif self.threadnum == 1: #Thread1
            while not self.ready1: continue
            self.ready1 = False
            self.current_edgearray = self.suppressed_edgearray1.copy()
            self.current_orientationarray = self.orientationarray1.copy()
            self.run1 = True
            self.frame = self.frame + 1
            self.threadnum = 2

        elif self.threadnum == 2: #Thread2
            while not self.ready2: continue
            self.ready2 = False
            self.current_edgearray = self.suppressed_edgearray2.copy()
            self.current_orientationarray = self.orientationarray2.copy()
            self.run2 = True
            self.frame = self.frame + 1
            self.threadnum = 0

        self.execute = False

    def _generate0(self):
      while True:
        while not self.run0: continue
        while self.Reader.framenum < self.frame: continue #Don't grab frame till ready
        currentframe = self.Reader.currentframe.copy()
        self.Reader.execute = True
        self.run0 = False
        edgearray0 = self.edgeGenerator0.detectEdges(currentframe)
        self.orientationarray0 = self.edgeGenerator0.computeOrientation(edgearray0)
        self.suppressed_edgearray0 = self.edgeGenerator0.edgesNms(edgearray0, self.orientationarray0)
        self.ready0 = True

    def _generate1(self):
      while True:
        while not self.run1: continue
        while self.Reader.framenum < self.frame or self.run0: continue #Don't grab frame till ready
        currentframe = self.Reader.currentframe.copy()
        self.Reader.execute = True
        self.run1 = False
        edgearray1 = self.edgeGenerator1.detectEdges(currentframe)
        self.orientationarray1 = self.edgeGenerator1.computeOrientation(edgearray1)
        self.suppressed_edgearray1 = self.edgeGenerator1.edgesNms(edgearray1, self.orientationarray1)
        self.ready1 = True

    def _generate2(self):
      while True:
        while not self.run2: continue
        while self.Reader.framenum < self.frame or self.run0 or self.run1: continue #Don't grab frame till ready
        currentframe = self.Reader.currentframe.copy()
        self.Reader.execute = True
        self.run2 = False
        edgearray2 = self.edgeGenerator2.detectEdges(currentframe)
        self.orientationarray2 = self.edgeGenerator2.computeOrientation(edgearray2)
        self.suppressed_edgearray2 = self.edgeGenerator2.edgesNms(edgearray2, self.orientationarray2)
        self.ready2 = True
