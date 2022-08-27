import timeit
import cv2
import numpy as np

IN_IMG = '../teste.JPG'

thresh_value = 245  # threshold to find white
blur_value = 50     # bloom smoothness
gain = 6            # bloom gain in intensity

### A ordem dos canais na imagem colorida é B-G-R!

img = cv2.imread (IN_IMG)
if img is None:
    print ('Erro abrindo %s' % IN_IMG)
rows, cols, channels = img.shape

## Convertendo imagens para o colorspace(canal) hsv (hue, saturation, value) como float
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
h, s, v = cv2.split(hsv)

# O melhor é que tenha baixa saturação e alto brilho (branco)
# Nesse caso, inverte-se a saturação (s) e multiplica pelo brilho (v)
sv = ((255 - s) * v / 255).clip(0, 255).astype(np.uint8)

# threshold
thresh = cv2.threshold(sv, thresh_value, 255, cv2.THRESH_BINARY)[1]

# blur and make 3 channels
blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=blur_value, sigmaY=blur_value)
blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

# Junta o blur e image usando gain no blur
result = cv2.addWeighted(img, 1, blur, gain, 0)

cv2.imshow('Flower Scene', cv2.resize(img, (940, 540)))

cv2.imshow('Flower Scene 2', cv2.resize(thresh, (940, 540)))
cv2.imshow('Flower Scene 3', cv2.resize(blur, (940, 540)))

cv2.imshow('Flower Scene 4', cv2.resize(result, (940, 540)))
cv2.waitKey()
cv2.destroyAllWindows()

