import timeit
import cv2
import numpy as np

IN_IMG = './teste.JPG'

### A ordem dos canais na imagem colorida é B-G-R!

img = cv2.imread (IN_IMG)
if img is None:
    print ('Erro abrindo %s' % IN_IMG)
rows, cols, channels = img.shape

# Convertendo para float32.
img = img.astype (np.float32) / 255

image_blurred = cv2.blur (img, ksize=(50, 50))

cv2.imshow('Flower Scene', cv2.resize(image_blurred, (940, 540)))
cv2.waitKey()
cv2.destroyAllWindows()

