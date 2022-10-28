

			#this file will use region growing algorithme for object detection

#Libirairies importation
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


#region growing function
class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
 
    def getX(self):
        return self.x
    def getY(self):
        return self.y
 
def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))
 
def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects
 
def regionGrow(img,seeds,thresh,p = 1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
 
        seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))
    return seedMark
 

#image importation and region growing testing
img = cv2.imread("feuille.png",0)
seeds = [Point(100,80)]
binaryImg = regionGrow(img,seeds,10)
cv2.imshow(' ',binaryImg)
cv2.waitKey(0)

#filtre for horizontal contours
newImage=img
filtre1 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
img1 = cv2.filter2D(img,-1,filtre1)
cv2.imshow('Contour vertical',img1)
cv2.waitKey(0)

#filtre for vertical contours
filtre2 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]]) 
img2 = cv2.filter2D(img,-1,filtre2)
cv2.imshow('Contour horizontal',img2)
cv2.waitKey(0) 

