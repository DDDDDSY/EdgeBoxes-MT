import cv2
import numpy as np
from classes import reader

file = "video.mp4"
modelfile = "model.yml.gz" #StructuredEdgeDetection model (generates edgemap)

video = reader(file)
video.read()

print("Loading model...")
edgeGenerator = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)

edgearray = np.zeros((video.height, video.width), dtype=np.float32) #Empty array for edgemap
print("Detecting...")
edgearray = edgeGenerator.detectEdges(video.frame)

cv2.imshow('image', edgearray)
cv2.waitKey(10000)
