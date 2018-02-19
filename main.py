import cv2
import numpy as np

imagefile = "image.jpg"
modelfile = "model.yml.gz" #StructuredEdgeDetection model (generates edgemap)

img = cv2.imread(imagefile)
imgarray = img.astype(np.float32) #Change type to be compatible with edgeDetect
h, w, d = imgarray.shape

def normalize(a): #Normalize to a float in [0,1]
    return np.float32(a/255)
imgarray = np.apply_along_axis(normalize, 0, imgarray) #Apply normalize function over array

print("Loading model...")
edgeGenerator = cv2.ximgproc.createStructuredEdgeDetection(model = modelfile)

edgearray = np.zeros((h, w), dtype=np.float32) #Empty array for edgemap
print("Detecting...")
edgearray = edgeGenerator.detectEdges(imgarray)

cv2.imshow('image', edgearray)
cv2.waitKey(10000)
