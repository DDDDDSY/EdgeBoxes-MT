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
        self.execute = False
        self.currentframe = self.frame.copy()
        self.framenum = self.framenum + 1
        _, self.frame = self.video_capture.read()
        if self.frame is None:
            exit("End of video file!")

        self.frame = self.frame.astype(np.float32) #Change type to be compatible with edgeDetect
        self.frame = np.divide(self.frame, 255.0) #Normalize to [0,1]

class generator:

    def __init__(self, modelfile, video):

        self.Reader = video #multimedia reader
        self.Reader.execute = True
        
        print("Loading model...")
        self.edgeGenerator = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)
        self.queue0 = Queue()
        self.queue1 = Queue()
        self.queue2 = Queue()
        
        ### INITIALIZE THREADS ###
        self.Thread0 = Process(target=self._generate,
                       args=(self.edgeGenerator, self.Reader.currentframe, self.queue0,),
                        daemon = True)
        print("Thread0 Initializing")
        self.Thread0.start()
        self.Reader.execute = True
        self.Thread0.join(5) #exit if hangs for more than 5 seconds
        
        self.Thread1 = Process(target=self._generate,
                       args=(self.edgeGenerator, self.Reader.currentframe, self.queue1,),
                       daemon = True)
        print("Thread1 Initializing")
        self.Thread1.start()
        self.Reader.execute = True
        self.Thread1.join(5) #exit if hangs for more than 5 seconds
        
        self.Thread2 = Process(target=self._generate,
                       args=(self.edgeGenerator, self.Reader.currentframe, self.queue2,),
                       daemon = True)
        print("Thread2 Initializing")
        self.Thread2.start()
        self.Reader.execute = True
        self.Thread2.join(5) #exit if hangs for more than 5 seconds
        ### END INITIALIZATION
        
        self.ready = False
        self.framenum = 3 #Frame count starts at 2 due to earlier inits
        self.execute = False #flags controlling this thread
        self.threadnum = 0

    def generate(self):
      while True: #continuously execute
        while not self.execute: continue

        if self.threadnum == 0: #Thread 0
        
            self.current_edgearray = self.queue0.get()
            self.current_orientationarray = self.queue0.get()
            self.execute = False

            self.framenum = self.framenum + 1
            
            while self.framenum > self.Reader.framenum: continue
            self.Thread0 = Process(target=self._generate,
                           args=(self.edgeGenerator, self.Reader.currentframe, self.queue0,),
                           daemon = True)
            self.Thread0.start()
            self.Reader.execute = True
            self.threadnum = 1
            
        elif self.threadnum == 1: #Thread 1
        
            self.current_edgearray = self.queue1.get()
            self.current_orientationarray = self.queue1.get()
            self.execute = False

            self.framenum = self.framenum + 1
            
            while self.framenum > self.Reader.framenum: continue
            self.Thread1 = Process(target=self._generate,
                           args=(self.edgeGenerator, self.Reader.currentframe, self.queue1,),
                           daemon = True)
            self.Thread1.start()
            self.Reader.execute = True
            self.threadnum = 2
            
        elif self.threadnum == 2: #Thread 2
        
            self.current_edgearray = self.queue2.get()
            self.current_orientationarray = self.queue2.get()
            self.execute = False

            self.framenum = self.framenum + 1
            
            while self.framenum > self.Reader.framenum: continue
            self.Thread2 = Process(target=self._generate,
                           args=(self.edgeGenerator, self.Reader.currentframe, self.queue2,),
                           daemon = True)
            self.Thread2.start()
            self.Reader.execute = True
            self.threadnum = 0

         
    def _generate(self, edgeGenerator, currentframe, q):
        edgearray = edgeGenerator.detectEdges(currentframe)
        orientationarray = edgeGenerator.computeOrientation(edgearray)
        suppressed_edgearray = edgeGenerator.edgesNms(edgearray, orientationarray)
        q.put(suppressed_edgearray)
        q.put(orientationarray)
