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
video.execute = True #Get frame ready

Generator = generator(modelfile, video)
generatorThread = threading.Thread(target=Generator.generate)
generatorThread.daemon = True
generatorThread.start()
Generator.execute = True

boxGenerator = cv2.ximgproc.createEdgeBoxes(maxBoxes = 50, alpha = 0.5)

frames = 0
total_fps = 0

try:
  while True:

    beginning = time.time() #For FPS calculations

    Generator.execute = True #start computing next edgemap and orientationmap
    while frames >= Generator.frame: continue #wait for next maps to be generated
    boxes = boxGenerator.getBoundingBoxes(Generator.current_edgearray, Generator.current_orientationarray)

    fps = 1/(time.time()-beginning)
    print("FPS: ", fps)

    frames = frames + 1
    total_fps = total_fps + fps

    cv2.imshow("image", draw_boxes(boxes, Generator.current_edgearray))
    cv2.waitKey(100)

except KeyboardInterrupt:
    exit(total_fps/frames)
