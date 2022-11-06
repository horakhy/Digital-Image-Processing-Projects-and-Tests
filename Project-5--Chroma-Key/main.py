import cv2
import sys
import numpy as np

path = 'img/3.bmp'
background = 'cow.jpg'
A1 = 0.5
A2 = 1.2

def masking(image):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    channel = lab[:, :, 1]

    mask = cv2.threshold(
        channel, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    cv2.imwrite('qaaaaa.png', mask)

    mask2 = channel*(-mask)
    mask2 = cv2.normalize(mask2, dst=None, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    masked_image = image
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            masked_image[x,y] = image[x,y] * (mask[x,y]/255)

    return masked_image, mask2

def removeGreenShades(masked_image, th):
    mlab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
    dst = cv2.normalize(mlab[:, :, 1], dst=None, alpha=0,
                        beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    threshold_value = 40

    dst_th = cv2.threshold(dst, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]
    mlab[:, :, 0][dst_th == 255] = 0

    img2 = cv2.cvtColor(mlab, cv2.COLOR_LAB2BGR)
    cv2.imwrite('img2.png', img2)

    img2[th == 0] = (0, 0, 0)

    return img2

def merge(mask, img_source, background):
    height = img_source.shape[1]
    width = img_source.shape[0]

    for y in range(height):
        for x in range(width):
            if mask[x,y] <= 30:
                img_source[x, y] = background[x, y]
            elif mask[x,y] > 30 and mask[x,y] < 130:
                img_source[x,y] = (img_source[x][y]) * (background[x][y])
    return img_source

image = cv2.imread(path)
if image is None:
    print('Erro abrindo a imagem.\n')
    sys.exit()
image = image.astype(np.uint8)
cv2.imwrite('00 - Original.png', image)
background = cv2.imread(background)
background = cv2.resize(background, (image.shape[1], image.shape[0]))
cv2.imwrite('background.jpg', background)

masked_image, mask = masking(image)
cv2.imwrite('01 - Mask.png', mask)
cv2.imwrite('02 - Masked Image.png', masked_image)

final_img = masked_image
# final_img = removeGreenShades(masked_image, mask)
cv2.imwrite('04 - removeGreenBorder.png', final_img)

final_image = merge(mask, final_img, background)

cv2.imwrite('05 - final.png', final_image)
