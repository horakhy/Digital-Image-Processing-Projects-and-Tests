import cv2
import sys
import numpy as np

path = 'img/1.bmp'
background = 'cow.jpg'
A1 = 0.5
A2 = 1.2

def masking(image):
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    channel = mask[:,:,1]
    th = cv2.threshold(channel, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    cv2.imwrite('02.png', th)

    masked_image = cv2.bitwise_and(image, image, mask = th)

    return mask, masked_image, th

def removeGreenBorder(masked_image, th):
    mlab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
    dst = cv2.normalize(mlab[:,:,1], dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    threshold_value = 127
    dst_th = cv2.threshold(dst, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]
    mlab2 = mlab.copy()
    for x in range(mlab.shape[2]):
        mlab[:,:,x][dst_th == 255] = 127
    img2 = cv2.cvtColor(mlab, cv2.COLOR_LAB2BGR)
    img2[th==0]=(255,255,255)

    return img2


def merge(mask,img_sorce,background):
 height = img_sorce.shape[1]
 width = img_sorce.shape[0]

 for y in range(height):
  for x in range(width):
   if mask[x,y] < 5:
    img_sorce[x,y] = background[x,y] # faz a troca binária se a mask for muito clara, oque representra sem muito verde
   elif mask[x,y] > 5 and mask[x,y] < 255:  
    img_sorce[x,y] = (((mask[x,y]/255))*background[x,y]) #aplica os pixels do backgrond de acordo com um peso, isso trata as bordos pois a mascara sofreu Blur.

 return img_sorce

image = cv2.imread(path)
if image is None:
    print('Erro abrindo a imagem.\n')
    sys.exit()
image = image.astype(np.uint8)
cv2.imwrite('00 - Original.png', image)

mask, masked_image, th = masking(image)
cv2.imwrite('01 - Mask.png', mask)
cv2.imwrite('02 - Masked Image.png', masked_image)

print('shape 0.jpg', mask.shape[0])
print('shape 1.jpg', mask.shape[1])
print('shape 2.jpg', mask.shape[2])

final_img = masked_image
# final_img = removeGreenBorder(masked_image, th)
cv2.imwrite('04 - removeGreenBorder.png', final_img)

background = cv2.imread(background)
background = cv2.resize(background,(image.shape[1],image.shape[0])) 
cv2.imwrite('background.jpg', background)

final_image = merge(th,final_img,background)


cv2.imwrite('05 - final.png', final_img)



