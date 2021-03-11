import os
from os.path import  join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

path = "Subset_train"
img_filenames=[]
img_filenames = [f for f in os.listdir(path)]

images = np.empty(len(img_filenames), dtype=object)        

for i in range(0,len(img_filenames)):
     images[i] = cv2.imread(join(path,img_filenames[i]))   
    

#resizing the images
resized_images = np.empty(len(img_filenames), dtype=object)
Resize_dimension = (320,320)
for j in range(0,len(images)):
     resized_images[j] = cv2.resize(images[j],Resize_dimension)   

#applying gaussian blur to each image
blurred_images = np.empty(len(img_filenames), dtype=object)
for i in range(0,len(resized_images)):
    blurred_images[i] = cv2.GaussianBlur(resized_images[i], (5, 5), 0)

#Converting to gray scale
grayscale_images = np.empty(len(img_filenames), dtype=object)    
for i in range(0,len(blurred_images)):
    grayscale_images[i] = cv2.cvtColor(blurred_images[i],cv2.COLOR_BGR2GRAY)

#applying threshold
kernel = np.ones((3, 3), np.uint8)
threshold_images = np.empty(len(img_filenames), dtype=object)    
for i in range(0,len(grayscale_images)):
    val, image =cv2.threshold(grayscale_images[i], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 
    morphed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
    dilated = cv2.dilate(morphed, kernel, iterations=3)
    transformed = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
    threshold_images[i] = transformed

#for i in range(0,len(img_filenames)):
#print(os.getcwd())        
    
    


cv2.imshow("resized image", resized_images[0])
print(resized_images[0].shape)
cv2.imshow("blurred image", blurred_images[0])
cv2.imshow("grayed image", grayscale_images[0])
cv2.imshow("thresholded image", threshold_images[0])
cv2.waitKey(0)
