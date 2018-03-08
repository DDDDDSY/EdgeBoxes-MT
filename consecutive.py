import cv2
import numpy as np
import threading
from multiprocessing import Process, Queue
import time

file = "video.mp4"
modelfile = "model.yml.gz" #StructuredEdgeDetection model (generates edgemap)

### Video Reader ###
video_capture = cv2.VideoCapture(file)

### Gen and Pred threads setup ###
print("Loading model...")
edgeGenerator = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)
boxGenerator = cv2.ximgproc.createEdgeBoxes(maxBoxes = 100, alpha = 0.65, beta = 0.75, minScore = 0.03)

### MAIN ###

frames = 0
total_fps = 0

bbeginning = time.time() #For final FPS

try:
  while True:

    beginning = time.time() #For FPS calculations
    
    _, frame = video_capture.read()
    
    if frame is None:
        print("End of Video File!")
        exit(frames/(time.time()-bbeginning))
    
    frame = frame.astype(np.float32)
    frame = np.divide(frame, 255.0)

    edgearray = edgeGenerator.detectEdges(frame)
    orientationarray = edgeGenerator.computeOrientation(edgearray)

    boxes = boxGenerator.getBoundingBoxes(edgearray, orientationarray)

    fps = 1/(time.time()-beginning)
    print("FPS: ", fps, "\n")

    frames = frames + 1
    total_fps = total_fps + fps

except KeyboardInterrupt:
    exit(frames/(time.time()-bbeginning))
