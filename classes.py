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

    def __init__(self, modelfile, video, num_threads):

        self.Reader = video #multimedia reader
        self.Reader.execute = True
        self.framenum = 0 #keep track of how many frames have been processed
        
        print("Loading model...")
        self.edgeGenerator = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)

        #Create a queue for each thread
        self.queue = list()
        for x in range(0, num_threads):
	        self.queue.append(Queue())
        
        ### INITIALIZE THREADS ###
        self.thread = list()
        for x in range(0, num_threads):
            self.thread.append(Process(target=self._generate,
                               args=(self.edgeGenerator,
                                     self.Reader.currentframe,
                                     self.queue[x],),
                               daemon = True))
            print("Thread", x, " Initializing")
            while self.framenum >= self.Reader.framenum: continue
            self.thread[x].start()
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
        
        while self.framenum >= self.Reader.framenum: continue #Wait for video reader
        
        self.thread[self.threadnum] = Process(target=self._generate,
                                              args=(self.edgeGenerator,
                                                    self.Reader.currentframe,
                                                    self.queue[self.threadnum],),
                                              daemon = True)
        self.thread[self.threadnum].start()
        self.framenum = self.framenum + 1
        self.Reader.execute = True
        self.threadnum = self.next_thread()

         
    def _generate(self, edgeGenerator, currentframe, q):
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
