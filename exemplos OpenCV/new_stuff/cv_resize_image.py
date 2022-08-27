import timeit
import cv2
import numpy as np

IN_IMG = '../teste.JPG'

### A ordem dos canais na imagem colorida é B-G-R!

img = cv2.imread (IN_IMG)
if img is None:
    print ('Erro abrindo %s' % IN_IMG)
rows, cols, channels = img.shape

# Convertendo para float32.
img = img.astype (np.float32) / 255

scale_percent = 60 # porcentagem do tamanho original
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

cv2.imshow('Flower Scene', img)
cv2.imshow('Flower Scene Resized',img_resized)
cv2.waitKey()
cv2.destroyAllWindows()

