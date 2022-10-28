
			#this file contaim image preparation and manipulation functions using python

#librairies importation
from PIL import Image
import PIL
import imageio
from matplotlib.pyplot import *
import math
import cv2
from matplotlib import pyplot as plt
import numpy as np


#image importation
image=Image.open("Chiffre_NG.jpg")
tab=np.array(image)
(ligne, colonne)=image.size
print(ligne,colonne)


#histogramme : a curve that denotes the distribution of the colors of filing in black and white

def histogramme(image):
    h = np.zeros(256)
    for j in range(ligne):
        for i in range(colonne):            
            valeur = tab[i,j]
            h[int(valeur)] += 1
    return h
h = histogramme(tab)
figure(figsize=(8,6))
plot(h)
axis([0,255,0,h.max()])
xlabel("valeur")
ylabel("Nombre")


#Dynamic cropping: for balance the density of black and white colors in the image

def recadrage(image,h):
    imin=np.min(image)
    imax=np.max(image)
    
    LUT = [0]*255
    
    for i in range(0,255):
        LUT[i] = math.ceil(255*(i-imin)/(imax-imin))
        
    s=image.shape
    for i in range(s[0]):
        for j in range(s[1]):
            image[i,j] = LUT[(image[i,j])]
    return image
    
# image_res = recadrage(tab, h)
# image_R = Image.fromarray(image_res)
# image_R.show()


#Thresholding: a function to define the appropriate threshold of the image

def seuillage(image,seuil):
    s=image.shape
    for i in range(s[0]):
        for j in range(s[1]): 
            if image[i,j] > seuil:
                image[i,j]=255
            else:
                image[i,j]=0
    return image

im_seuil=seuillage(tab, 170)     
image_s = Image.fromarray(im_seuil)
image_s.show()   



#Convolution: convulution 2d of a filtre on the image

img = cv2.imread("Brain.png")
kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
#kernel= np.array([[1,2,1],[2,4,2],[1,2,1]])/16
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
img2=cv2.medianBlur(img, 5)

cv2.imshow('mid',img2)
cv2.waitKey(0)


#Dilatation: for minimising the white pixels in the image

img = cv2.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(img,kernel,iterations = 1)
cv2.imshow('mid',dilation)
cv2.waitKey(0)
