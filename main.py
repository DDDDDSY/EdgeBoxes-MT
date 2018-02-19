import cv2
import numpy as np
import threading
import time

from classes import reader

file = "video.mp4"
modelfile = "model.yml.gz" #StructuredEdgeDetection model (generates edgemap)

video = reader(file)
videoThread = threading.Thread(target=video.read)
videoThread.start()

print("Loading model...")
edgeGenerator = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)

boxGenerator = cv2.ximgproc.createEdgeBoxes()

frames = 0
total_fps = 0

try:
  while True:

    beginning = time.time() #For FPS calculations

    video.execute = True #Start reading next frame

    edgearray = np.zeros((video.height, video.width), dtype=np.float32) #Empty array for edgemap
    edgearray = edgeGenerator.detectEdges(video.currentframe) #process current frame
    orientationarray = np.zeros((video.height, video.width), dtype=np.float32) #Empty array for orientation emap
    orientationarray = edgeGenerator.computeOrientation(edgearray)

    boxes = boxGenerator.getBoundingBoxes(edgearray, orientationarray)

    fps = 1/(time.time()-beginning)
    print("FPS: ", fps)

    frames = frames + 1
    total_fps = total_fps + fps

except KeyboardInterrupt:
    exit(total_fps/frames)
