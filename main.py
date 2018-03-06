import cv2
import numpy as np
import threading
from multiprocessing import Process, Queue
import time

from classes import reader, generator, predictor
from functions import draw_boxes

### CLI arguments parsing ###
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("num_threads", help="Specify number of generation threads")
parser.add_argument("ratio", help="Specify number of gen threads per pred thread, ie. 2")
parser.add_argument("visualize", help="Display video? y/n")
args = parser.parse_args()

file = "video.mp4"
modelfile = "model.yml.gz" #StructuredEdgeDetection model (generates edgemap)

### video reading thread setup ###
video = reader(file)
videoThread = threading.Thread(target=video.read)
videoThread.daemon = True
videoThread.start()

### Gen and Pred threads setup ###
Generator = generator(modelfile, video, int(args.num_threads))
generatorThread = threading.Thread(target=Generator.generate)
generatorThread.daemon = True
generatorThread.start() #starts with first maps ready

Predictor = predictor(Generator, int(args.ratio))
predictorThread = threading.Thread(target = Predictor.predict)
predictorThread.daemon = True
predictorThread.start()

### MAIN ###

frames = 0
total_fps = 0

bbeginning = time.time() #For final FPS

try:
  while True:

    beginning = time.time() #For FPS calculations
    
    boxes = Predictor.boxes.get()

    fps = 1/(time.time()-beginning)
    print("FPS: ", fps, "\n")

    if args.visualize == 'y':
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
