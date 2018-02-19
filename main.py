import cv2
import numpy as np
import threading
import time

from classes import reader

file = "video.mp4"
modelfile = "model.yml.gz" #StructuredEdgeDetection model (generates edgemap)

video = reader(file)

print("Loading model...")
edgeGenerator = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)

frames = 0
total_fps = 0

try:
  while True:

    beginning = time.time() #For FPS calculations

    video.read()
    edgearray = np.zeros((video.height, video.width), dtype=np.float32) #Empty array for edgemap
    edgearray = edgeGenerator.detectEdges(video.frame)

    fps = 1/(time.time()-beginning)
    print("FPS: ", fps)

    frames = frames + 1
    total_fps = total_fps + fps

except KeyboardInterrupt:
    exit(total_fps/frames)
