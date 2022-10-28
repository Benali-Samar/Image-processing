

			# this file will descrie image manipulation using python
			
#librairies importation
from PIL import Image, ImageOps
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

#image importation
im=Image.open("Lena.jpg")
tab=np.array(im)
im.show()
M = misc.imread('Lena.jpg')
M.shape
print(M.dtype)

#matrice partition
rouge = M[:,:,0]
vert = M[:,:,1]
bleu = M[:,:,2]

#red filtre

def Rouge(image):
    (c,l) =image.size
    imagearrivee=Image.new('RGB',(c,l))
    for x in range(c):
        for y in range(l):
            pixel=image.getpixel((x,y))
            p=(pixel[2],0,0)
            imagearrivee.putpixel((x,y),p)
    imagearrivee.save("lena-rouge.JPG")       
Rouge(im)


#Blue filtre 
def Bleu(image):
    (c,l) =image.size
    imagearrivee=Image.new('RGB',(c,l))
    for x in range(c):
        for y in range(l):
            pixel=image.getpixel((x,y))
            p=(0,0,pixel[2])
            imagearrivee.putpixel((x,y),p)
    imagearrivee.save("lena-bleu.JPG")    
Bleu(im)


#green filtre
def Vert(image):
    (c,l) =image.size
    imagearrivee=Image.new('RGB',(c,l))
    for x in range(c):
        for y in range(l):
            pixel=image.getpixel((x,y))
            p=(0,pixel[2],0)
            imagearrivee.putpixel((x,y),p)
    imagearrivee.save("lena-vert.JPG")    
Vert(im)


#grey filtre 
def Gris(image):
    (c,l) =image.size
    imagearrivee=Image.new('RGB',(c,l))
    for x in range(c):
        for y in range(l):
            pixel=image.getpixel((x,y))
            gris= int((0.299*pixel[2]+0.587*pixel[2]+0.114*pixel[2]))
            p=(gris,gris,gris)
            imagearrivee.putpixel((x,y),p)
    imagearrivee.save("lena-gris.JPG")
Gris(im)


#image inversion
im=Image.open("Lena.jpg")
tab=np.array(im)
im.show()
def invert(image):
    (c,l) =image.size
    imagearrivee=Image.new('RGB',(c,l))
    for x in range(c):
        for y in range(l):
            pixel=image.getpixel((x,y))
            p=(255-pixel[1],255-pixel[1],255-pixel[1])
            imagearrivee.putpixel((x,y),p)
    imagearrivee.save("lena-invert1.JPG") 
invert(im)


#Black and white
echiquier_array = np.zeros([200, 200], dtype = np.uint8)
for x in range(200):
    for y in range(200):
        if (x % 50) // 25 == (y % 50) // 25:
            echiquier_array[x, y] = 0
        else:
            echiquier_array[x, y] = 255
plt.imshow(echiquier_array, cmap='Greys_r')
plt.show()  


#liminosity controle
def lum(image):
    (c,l) =image.size
    imagearrivee=Image.new('RGB',(c,l))
    b=3
    for x in range(c):
        for y in range(l):
            pixel=image.getpixel((x,y))
            
            p=(pixel[1]-b,pixel[1]-b,pixel[1]-b)
            
            imagearrivee.putpixel((x,y),p)
    imagearrivee.save("lena-lum.jpg") 
lum(im)


#contraste controle
def con(image):
    (c,l) =image.size
    imagearrivee=Image.new('RGB',(c,l))
    a=60
    for x in range(c):
        for y in range(l):
            pixel=image.getpixel((x,y))
            
            p=(pixel[1]-a,pixel[1]-a,pixel[1]-a)
            
            imagearrivee.putpixel((x,y),p)
    imagearrivee.save("lena-con.jpg") 
con(im)

