import cv2
import numpy as np
from multiprocessing import Process, Queue

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
        self.edgeGenerator = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)
        self.queue0 = Queue()

        self.frame = 0 #Frame count
        self.execute = False #flags controlling this thread
        self.threadnum = 0
        
        self.Reader.execute = True

    def generate(self):
      while True: #continuously execute
        while not self.execute: continue

        Thread0 = Process(target=self._generate,
                          args=(self.edgeGenerator, self.Reader.currentframe, self.queue0,),
                          daemon = True)
        print("Thread0 Start")
        Thread0.start()
        self.Reader.execute = self.queue0.get()
        self.current_edgearray = self.queue0.get()
        self.current_orientationarray = self.queue0.get()
        Thread0.join(5) #exit if hangs for more than 5 seconds

        self.execute = False
        
    def _generate(self, edgeGenerator, currentframe, q):
        edgearray = edgeGenerator.detectEdges(currentframe)
        q.put(True) #Execute next frame read
        orientationarray = edgeGenerator.computeOrientation(edgearray)
        suppressed_edgearray = edgeGenerator.edgesNms(edgearray, orientationarray)
        q.put(suppressed_edgearray)
        q.put(orientationarray)