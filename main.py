import cv2
import numpy as np

imagefile = "image.jpg"
modelfile = "model.yml.gz" #StructuredEdgeDetection model (generates edgemap)

img = cv2.imread(imagefile)
imgarray = img.astype(np.float32) #Change type to be compatible with edgeDetect

def normalize(a):
    return np.float32(a/255)

imgarray = np.apply_along_axis(normalize, 0, imgarray)
h, w, d = imgarray.shape

print("Loading model...")
edgeGenerator = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)

edgearray = np.zeros((h, w), dtype=np.float32) #Empty array for edgemap
print("Detecting...")
edgearray = edgeGenerator.detectEdges(imgarray)

cv2.imshow('image', edgearray)
cv2.waitKey(10000)
