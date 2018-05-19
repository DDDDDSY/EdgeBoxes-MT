import cv2
import numpy as np
import threading
from multiprocessing import Process, Queue
import time

from classes import readerNetwork, generator, predictor
reader = readerNetwork
from functions import draw_boxes

### CLI arguments parsing ###
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("gen_threads", help="Specify number of generation threads")
parser.add_argument("pred_threads", help="Specify number of prediction threads")
parser.add_argument("visualize", help="Display video? y/n")
args = parser.parse_args()

#file = "input.mp4"
modelfile = "model.yml.gz" #StructuredEdgeDetection model (generates edgemap)

### video reading thread setup ###
video = reader(9001)
videoThread = threading.Thread(target=video.read)
videoThread.daemon = True
videoThread.start()

### Gen and Pred threads setup ###
Generator = generator(modelfile, video, int(args.gen_threads))
generatorThread = threading.Thread(target=Generator.generate)
generatorThread.daemon = True
generatorThread.start() #starts with first maps ready

Predictor = predictor(Generator, int(args.pred_threads))
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
    edgearray = Predictor.boxes.get()
    video_frame = Predictor.boxes.get()

    if boxes is None:
        exit()
        #with open("fps.log", "a") as myfile: #Write fps to file for logging
        #    myfile.write(str(args.gen_threads)+
        #                 " "+str(args.pred_threads)+
        #                 " "+str(256/(time.time()-bbeginning))+
        #                 "\n")
        #exit(256/(time.time()-bbeginning))

    fps = 1/(time.time()-beginning)
    print("FPS: ", int(fps))

    if args.visualize == 'y':
      frame = draw_boxes(boxes, edgearray)
      cv2.imshow('image', frame)
      cv2.imshow('edgemap', edgearray)
      cv2.imshow('original', video_frame)
      cv2.waitKey(10)

    frames = frames + 1
    total_fps = total_fps + fps

except KeyboardInterrupt:
    exit()
