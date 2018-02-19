import cv2
import numpy as np
import threading
import time

from classes import reader
from functions import draw_boxes

file = "video.mp4"
modelfile = "model.yml.gz" #StructuredEdgeDetection model (generates edgemap)

video = reader(file)
videoThread = threading.Thread(target=video.read)
videoThread.start()
video.execute = True #Get frame ready

print("Loading model...")
edgeGenerator = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)

boxGenerator = cv2.ximgproc.createEdgeBoxes()

frames = 0
total_fps = 0

try:
  while True:

    beginning = time.time() #For FPS calculations

    edgearray = edgeGenerator.detectEdges(video.currentframe) #process current frame
    video.execute = True #read for next loop iteration
    orientationarray = edgeGenerator.computeOrientation(edgearray)
    suppressed_edgearray = edgeGenerator.edgesNms(edgearray, orientationarray)

    boxes = boxGenerator.getBoundingBoxes(suppressed_edgearray, orientationarray)

    fps = 1/(time.time()-beginning)
    print("FPS: ", fps)

    frames = frames + 1
    total_fps = total_fps + fps

    frame = draw_boxes(boxes, edgearray)
    cv2.imshow('image', frame)
    cv2.waitKey(1)

except KeyboardInterrupt:
    exit(total_fps/frames)
