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
        self.Reader.execute = True
        
        print("Loading model...")
        self.edgeGenerator = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)
        self.queue0 = Queue()
        self.queue1 = Queue()
        
        
        ### INITIALIZE THREADS ###
        self.Thread0 = Process(target=self._generate,
                       args=(self.edgeGenerator, self.Reader.currentframe, self.queue0,),
                        daemon = True)
        print("Thread0 Initializing")
        self.Thread0.start()
        self.Reader.execute = self.queue0.get() 
        self.current_edgearray0 = self.queue0.get()
        self.current_orientationarray0 = self.queue0.get()
        self.Thread0.join(5) #exit if hangs for more than 5 seconds
        
        self.Thread1 = Process(target=self._generate,
                       args=(self.edgeGenerator, self.Reader.currentframe, self.queue1,),
                       daemon = True)
        print("Thread1 Initializing")
        self.Thread1.start()
        self.Reader.execute = self.queue1.get() 
        self.current_edgearray1 = self.queue1.get()
        self.current_orientationarray1 = self.queue1.get()
        self.Thread1.join(5) #exit if hangs for more than 5 seconds
        ### END INITIALIZATION
        
        
        self.frame = 1 #Frame count
        self.execute = False #flags controlling this thread
        self.threadnum = 1 #Start on thread 1 since thread 0 has data

    def generate(self):
      while True: #continuously execute
        while not self.execute: continue

        if self.threadnum == 0: #Thread 0
        
            self.current_edgearray = self.current_edgearray0
            self.current_orientationarray = self.current_orientationarray0
            self.execute = False
            
            self.Thread0.join(5) #exit if hangs for more than 5 seconds
            self.Thread0 = Process(target=self._generate,
                           args=(self.edgeGenerator, self.Reader.currentframe, self.queue0,),
                           daemon = True)
            print("Thread0 Start")
            self.Thread0.start()
            self.Reader.execute = self.queue0.get()
            self.current_edgearray0 = self.queue0.get()
            self.current_orientationarray0 = self.queue0.get()
            self.threadnum = 1

        elif self.threadnum == 1: #Thread 1
        
            self.current_edgearray = self.current_edgearray1
            self.current_orientationarray = self.current_orientationarray1
            self.execute = False
            
            self.Thread1.join(5) #exit if hangs for more than 5 seconds
            self.Thread1 = Process(target=self._generate,
                           args=(self.edgeGenerator, self.Reader.currentframe, self.queue1,),
                           daemon = True)
            print("Thread1 Start")
            self.Thread1.start()
            self.Reader.execute = self.queue1.get()
            self.current_edgearray1 = self.queue1.get()
            self.current_orientationarray1 = self.queue1.get()
            self.threadnum = 0

    def _generate(self, edgeGenerator, currentframe, q):
        edgearray = edgeGenerator.detectEdges(currentframe)
        q.put(True) #Execute next frame read
        orientationarray = edgeGenerator.computeOrientation(edgearray)
        suppressed_edgearray = edgeGenerator.edgesNms(edgearray, orientationarray)
        q.put(suppressed_edgearray)
        q.put(orientationarray)