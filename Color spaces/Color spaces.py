

				#Color spaces for images

#librairies importation			
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 


#presentation of 4 color spaces on F& image
 
img = cv2.imread("F1.jpg")
plt.imshow(img)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

cv2.namedWindow('hsv', cv2.WINDOW_NORMAL)
cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)
cv2.namedWindow('lab', cv2.WINDOW_NORMAL)
cv2.namedWindow('luv', cv2.WINDOW_NORMAL)

cv2.imshow('lab',lab)
cv2.imshow('luv',luv)
cv2.imshow('hsv',hsv)
cv2.imshow('rgb',rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()

#Kmeans testing for color extracting!


img1 = cv2.imread("F1.jpg")
img2 = cv2.imread("F2.jpg")

image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.imshow(image)
Vecteur = image.reshape((-1,3)) 
Vecteur = np.float32(Vecteur)
criteres= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,200, 5) 
k = 2
retval, labels, centers = cv2.kmeans(Vecteur, k, None, criteres, 10, cv2.KMEANS_RANDOM_CENTERS) 
centers = np.uint8(centers) 
segmented_data = centers[labels.flatten()] 
Nouvelle_image1= segmented_data.reshape((image.shape)) 

plt.imshow(Nouvelle_image1)



image = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imshow(image)
Vecteur = image.reshape((-1,3)) 
Vecteur = np.float32(Vecteur)
criteres= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,200, 5) 
k = 2
retval, labels, centers = cv2.kmeans(Vecteur, k, None, criteres, 10, cv2.KMEANS_RANDOM_CENTERS) 
centers = np.uint8(centers) 
segmented_data = centers[labels.flatten()] 
Nouvelle_image2= segmented_data.reshape((image.shape)) 

plt.imshow(Nouvelle_image2)
