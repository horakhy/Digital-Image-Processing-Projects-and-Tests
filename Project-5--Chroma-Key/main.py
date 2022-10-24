from pickletools import uint8
import cv2
import sys
from cv2 import COLOR_BGR2GRAY
import numpy as np

path = 'img/3.bmp'
background = 'cow.jpg'
A1 = 0.9
A2 = 0.2

def masking(image):
    (B,G,R) = cv2.split(image)
    mask = np.copy(image)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = (1 - A1*(G - A2*B))
    # (t,mask) = cv2.threshold(mask, len(mask), 1, 1, cv2.THRESH_TRUNC)
    # (t,mask) = cv2.threshold(-1*mask, len(mask), 0, 0, cv2.THRESH_TRUNC)
    alpha = mask


    alpha = alpha.astype (np.uint8) / 255

    for x in range(alpha.shape[0]):
            for y in range(alpha.shape[1]):
                if(alpha[x][y] < 0.3):
                    alpha[x][y] = 0

    print('alpha:', alpha)

    return alpha

image = cv2.imread(path)
if image is None:
    print('Erro abrindo a imagem.\n')
    sys.exit()
image = image.astype(np.uint8)
cv2.imwrite('00 - Original.png', image)

mask = masking(image)
cv2.imwrite('01 - Mask.png', mask* 255)

masked_image = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
masked_image = masked_image.astype(np.float64)
# masked_image*=mask
for x in range(masked_image.shape[0]):
    for y in range(masked_image.shape[1]):
        for c in range(masked_image.shape[2]):
            masked_image[x][y][c] = masked_image[x][y][c]*mask[x][y]

cv2.imwrite('02 - Masked Image.png', masked_image)

background = cv2.imread(background)
background = background.astype(np.float64)
cropped_background = background[0:mask.shape[0], 0:mask.shape[1]]
# cropped_background[mask == 0] = [0, 0, 0]
mask = 1 - mask
print('alpha:', mask)

for x in range(cropped_background.shape[0]):
    for y in range(cropped_background.shape[1]):
        for c in range(cropped_background.shape[2]):
            cropped_background[x][y][c] = cropped_background[x][y][c]*(mask[x][y])
cv2.imwrite('03 - Background.png', cropped_background)

final_img = cropped_background + masked_image
cv2.imwrite('04 - final.png', final_img)



