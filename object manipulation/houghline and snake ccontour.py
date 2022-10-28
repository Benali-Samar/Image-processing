

			#actif contour: snake contour & houghline testing using pyton and opencv

#librairies importation
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

#image importation
image = cv2.imread("feuille.png")
plt.imshow(image)

#snake contour
image = rgb2gray(image)
s = np.linspace(0, 2*np.pi, 1000)
r =120 + 100*np.sin(s)
c = 120 + 100*np.cos(s)
init = np.array([r, c]).T
snake = active_contour(gaussian(image,3, preserve_range=False),
init, alpha=0.1, beta=10, gamma=0.001, max_iterations=8000)
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(image, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, image.shape[1], image.shape[0], 0])
plt.show()

#houghline
image = cv2.imread("Road.png")
plt.imshow(image)
grayscale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(grayscale,200,500)
edges = cv2.Canny(grayscale, 200, 500)
lines = cv2.HoughLinesP(edges,1.8,np.pi/250,70,np.array([]) ,60,50)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(image, (x1, y1), (x2, y2),color=(100,220,20),thickness=3)

plt.imshow(image)
