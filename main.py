import cv2
import numpy as np
import threading
from multiprocessing import Process, Queue
import time

from classes import reader, generator, predictor
from functions import draw_boxes

file = "video.mp4"
modelfile = "model.yml.gz" #StructuredEdgeDetection model (generates edgemap)

video = reader(file)
videoThread = threading.Thread(target=video.read)
videoThread.daemon = True
videoThread.start()

num_threads = 6
Generator = generator(modelfile, video, num_threads)
generatorThread = threading.Thread(target=Generator.generate)
generatorThread.daemon = True
generatorThread.start() #starts with first maps ready

Predictor = predictor(Generator)
predictorThread = threading.Thread(target = Predictor.predict)
predictorThread.daemon = True
predictorThread.start()

frames = 0
total_fps = 0

bbeginning = time.time() #For final FPS

try:
  while True:

    beginning = time.time() #For FPS calculations
    
    boxes = Predictor.boxes.get()

    fps = 1/(time.time()-beginning)
    print("FPS: ", fps, "\n")

    visualize = False
    if visualize:
      frame = draw_boxes(boxes, Generator.current_edgearray)
      cv2.imshow('image', frame)
      cv2.imshow('edgemap', Generator.current_edgearray)
      cv2.waitKey(10)

    frames = frames + 1
    total_fps = total_fps + fps

    if video.frame is None:
        print("End of Video File!")
        exit(frames/(time.time()-bbeginning))

except KeyboardInterrupt:
    exit(frames/(time.time()-bbeginning))
