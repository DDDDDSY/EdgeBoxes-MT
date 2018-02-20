import cv2
import numpy as np
import threading
import time

from classes import reader, generator
from functions import draw_boxes

file = "video.mp4"
modelfile = "model.yml.gz" #StructuredEdgeDetection model (generates edgemap)

video = reader(file)
videoThread = threading.Thread(target=video.read)
videoThread.daemon = True
videoThread.start()
video.execute = True #starts with a frame ready

Generator = generator(modelfile, video)
generatorThread = threading.Thread(target=Generator.generate)
generatorThread.daemon = True
generatorThread.start() #starts with first maps ready

boxGenerator = cv2.ximgproc.createEdgeBoxes(maxBoxes = 1000,
                                            alpha = 0.65,
                                            beta = 0.75,
                                            minScore = 0.02)

frames = 0
total_fps = 0

Generator.execute = True

try:
  while True:

    beginning = time.time() #For FPS calculations

    if Generator.execute: print("waiting...")
    while Generator.execute: continue
    beginning = time.time()
    boxes = boxGenerator.getBoundingBoxes(Generator.current_edgearray, Generator.current_orientationarray)
    #box[0] x1, box[1] x2, box[2] width, box[3] height
    print("BEx: ", round(time.time()-beginning, 3))
    print("BBs: ", len(boxes))
    Generator.execute = True

    fps = 1/(time.time()-beginning)
    print("FPS: ", fps, "\n")

    visualize = False
    if visualize:
      frame = draw_boxes(boxes, Generator.current_edgearray)
      cv2.imshow('image', frame)
      cv2.waitKey(10)
    else:
      cv2.imshow('image', Generator.current_edgearray)
      cv2.waitKey(10)

    frames = frames + 1
    total_fps = total_fps + fps

except KeyboardInterrupt:
    exit(total_fps/frames)
