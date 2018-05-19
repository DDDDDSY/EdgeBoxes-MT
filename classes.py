import cv2
import numpy as np
from multiprocessing import Process, Queue
import math

#Pull feed from udp stream
class readerNetwork:

    def __init__(self, port):

        #Assumes video stream is already started
        self.video_capture = cv2.VideoCapture("udp://127.0.0.1:"+str(port))
        self.execute = False

        self.framenum = 0

    def read(self):
      while True:

        _, self.frame = self.video_capture.read()
        self.framenum = self.framenum + 1

        if self.frame is None:
            self.currentframe = None
            exit("End of video stream!")

        self.frame = self.frame.astype(np.float32) #Change type to be compatible with edgeDetect
        self.frame = np.divide(self.frame, 255.0) #Normalize to [0,1]
        self.currentframe = self.frame.copy()


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
            self.currentframe = None
            exit("End of video file!")

        self.frame = self.frame.astype(np.float32) #Change type to be compatible with edgeDetect
        self.frame = np.divide(self.frame, 255.0) #Normalize to [0,1]

        
class generator:

    def __init__(self, modelfile, video, num_threads):

        self.Reader = video #multimedia reader
        self.Reader.execute = True
        self.framenum = 0 #keep track of how many frames have been processed

        print("Loading model...")
        self.edgeGenerator = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)

        #Create an output and input queue for each thread
        self.queue = list()
        self.queuein = list()
        for x in range(0, num_threads):
            self.queue.append(Queue())
            self.queuein.append(Queue())
            
        ### INITIALIZE THREADS ###
        self.thread = list()
        for x in range(0, num_threads):
            if self.framenum >= self.Reader.framenum: print("gen waiting...")
            while self.framenum >= self.Reader.framenum: continue #wait for frame
            self.thread.append(Process(target=self._generate,
                               args=(self.edgeGenerator,
                                     self.queuein[x],
                                     self.queue[x],),
                               daemon = True))
            print("Generation Thread", x, " Initializing")
            self.thread[x].start()
            self.queuein[x].put(self.Reader.currentframe)
            self.framenum = self.framenum + 1
            self.Reader.execute = True
        
        self.num_threads = num_threads
        self.ready = False
        self.execute = False #flags controlling this thread
        self.threadnum = 0

    def generate(self):
      while True: #continuously execute
        while not self.execute: continue

        self.current_edgearray = self.queue[self.threadnum].get()
        self.current_orientationarray = self.queue[self.threadnum].get()
        self.execute = False
        
        if self.framenum >= self.Reader.framenum: print("gen waiting...")
        while self.framenum >= self.Reader.framenum and self.Reader.currentframe is not None: continue #Wait for video reader
        
        if self.Reader.currentframe is None: #Exit when video is over
            self.current_edgearray = None
            self.current_orientationarray = None
            print("Generate done...")
            exit()
        
        self.queuein[self.threadnum].put(self.Reader.currentframe)
        self.framenum = self.framenum + 1
        self.Reader.execute = True
        self.threadnum = self.next_thread()

         
    def _generate(self, edgeGenerator, qin, q):
      while True:
        currentframe = qin.get()
        edgearray = edgeGenerator.detectEdges(currentframe)
        orientationarray = edgeGenerator.computeOrientation(edgearray)
        suppressed_edgearray = edgeGenerator.edgesNms(edgearray, orientationarray)
        q.put(suppressed_edgearray)
        q.put(orientationarray)
        
    def next_thread(self):
        if self.threadnum == self.num_threads - 1:
            return 0
        else:
            return self.threadnum + 1



### Class to generate boxes in a multithreaded environment. Transparent to main ###
class predictor:

    def __init__(self, generator, num_threads):

        self.generator = generator
        self.generator.execute = True #prepare current_edge*
    
        self.num_threads = num_threads
        
        self.framenum = 0 #keep track of how many frames have been processed
        
        self.boxGenerator = cv2.ximgproc.createEdgeBoxes(maxBoxes = 100,
                                                    alpha = 0.65,
                                                    beta = 0.75,
                                                    minScore = 0.03)

        #Create a queue for each thread
        self.queue = list()
        self.queuein = list()
        for x in range(0, self.num_threads):
            self.queue.append(Queue())
            self.queuein.append(Queue())

        ### INITIALIZE THREADS ###
        self.thread = list()
        for x in range(0, self.num_threads):
            if self.generator.execute: print("pred waiting...")
            while self.generator.execute: continue #wait for edgemap
            self.thread.append(Process(target=self._predict,
                               args=(self.boxGenerator,
                                     self.queuein[x],
                                     self.queue[x],),
                               daemon = True))
            print("Prediction Thread", x, "Initializing")
            self.thread[x].start()
            self.queuein[x].put(self.generator.current_edgearray)
            self.queuein[x].put(self.generator.current_orientationarray)
            self.generator.execute = True
            self.framenum = self.framenum + 1
        
        self.threadnum = 0
        self.boxes = Queue()

    
    def predict(self):
      while True:
        
        self.boxes.put(self.queue[self.threadnum].get()) #boxes
        self.boxes.put(self.queue[self.threadnum].get()) #edgemap
        
        if self.generator.execute: print("pred waiting...")
        while self.generator.execute and self.generator.current_edgearray is not None : continue #Wait for generator
        
        if self.generator.current_edgearray is None: #exit once video is done
            self.boxes.put(None) #boxes
            self.boxes.put(None) #edgemap
            print("Prediction done...")
            exit()
        
        self.queuein[self.threadnum].put(self.generator.current_edgearray)
        self.queuein[self.threadnum].put(self.generator.current_orientationarray)
        self.generator.execute = True
        self.framenum = self.framenum + 1
        self.threadnum = self.next_thread()

        
    def _predict(self, boxgenerator, qin, q):
      while True:
        edgearray = qin.get()
        orientationarray = qin.get()
        boxes = boxgenerator.getBoundingBoxes(edgearray, orientationarray)
        q.put(boxes)
        q.put(edgearray)
        
    def next_thread(self):
        if self.threadnum == self.num_threads - 1:
            return 0
        else:
            return self.threadnum + 1
