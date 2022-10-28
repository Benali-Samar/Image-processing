
			#texte image manipulation using python

#libraries importation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
get_ipython().run_line_magic('matplotlib', 'inline')


#image importation
image = cv2.imread("Texte_NG.jpg")   	
	#text image but bad quality!! so we need to devide the object from the background using kmeans algorithme
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)


# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1,3))
pixel_vals = np.float32(pixel_vals)

#the below line of code defines the criteria for the algorithm to stop running,
#which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
#becomes 85%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
 
# then perform k-means clustering wit h number of clusters defined as 2
#also random centres are initially choosed for k-means clustering
k = 2
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
 
# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
 
# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
plt.imshow(segmented_image)


# Globale thresholding for the image

tab=np.array(image)
plt.hist(tab.flatten(),bins=255,density =False, alpha=0.5,edgecolor='black', color='blue')
plt.show()
th=20
img1_modified=image.copy()
img1_modified[:,:1]=np.where(image[:,:,1]>th,image[:,:,1],0)
tab2=np.array(img1_modified)
plt.hist(tab2.flatten(),bins=255,density =False, alpha=0.5,edgecolor='black', color='blue')
plt.show()

fenetre='Image binaire OTSU'
cv.namedWindow(fenetre)
thresh,imgbin=cv.threshold(img,0,255,cv.THRESH_OTSU)
cv.imshow(fenetre,imgbin)
print('Valeur seuil OTSU', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Globale thresholding using a cursor

def seuillage(value):
    global imgbin
    print (value)
    ret,imgbin = cv.threshold(img,value,255,cv.THRESH_BINARY)

img = cv.imread("Texte_NG.jpg",cv.IMREAD_GRAYSCALE)
fenetre = 'Image binaire'
cv.namedWindow(fenetre)

# create trackbars for color change
cv.createTrackbar('Thresh',fenetre,0,255,seuillage)
seuillage(0)
while(True):
    cv.imshow(fenetre,imgbin)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break;
cv.destroyAllWindows()
plt.imshow(img1_modified,cmap='gray')

# Adaptative thresholding for every zone in the image : more spécification
#this gives the best result for text image thresholding!!

image1 = cv2.imread('Texte_NG.jpg')
img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 199, 5)
thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 199, 5)

cv2.imshow('Adaptive Mean', thresh1)
cv2.imshow('Adaptive Gaussian', thresh2)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()



