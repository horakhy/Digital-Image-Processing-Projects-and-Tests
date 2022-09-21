#===============================================================================
# Exemplo: blur de uma imagem.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'GT2.BMP'
SIGMA_LIMIT = 5
KERNEL = 5
SIGMA_MULTIPLIER = 2
BLUR_SUM_COUNTER = SIGMA_LIMIT

def gaussianBlur(img):
    sigma = SIGMA_MULTIPLIER
    img_to_be_blurred = img.copy()

    for sigma in range(SIGMA_LIMIT):
        img_to_be_blurred += cv2.GaussianBlur(img_to_be_blurred, (KERNEL,KERNEL), sigma)
        sigma*=SIGMA_MULTIPLIER
    
    return img_to_be_blurred

def boxBlur(img):
    img_to_be_blurred = img.copy()
    for x in range(BLUR_SUM_COUNTER):
        img_to_be_blurred += cv2.blur(img_to_be_blurred, (KERNEL,KERNEL))
        x+=1
    return img_to_be_blurred

def merge_img(img, blurred_bright_pass, alpha):
    
    return (img*alpha)+(blurred_bright_pass*(1 - alpha))

#===============================================================================

def main ():

    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()
    img = img.reshape ((img.shape [0], img.shape [1], 3))

    img = img.astype (np.float32) / 255.0
    alpha = 0.95

    ## Apenas o canal de luminosidade
    lightness = img[:,:,1]
    
    ## Criando imagem da máscara contendo as fontes de luz
    light_mask = cv2.inRange(lightness, 0.5, 1.0)

    start_time = timeit.default_timer()
    
    img_bright_pass = cv2.bitwise_and(img, img, mask = light_mask)

    blurred_bright_pass = gaussianBlur(img_bright_pass)
    img_out = merge_img(img, blurred_bright_pass, alpha)
    print("Tempo: %f" % (timeit.default_timer() - start_time))

    # cv2.imshow("01 - light_mask", light_mask)
    # cv2.imwrite("01 - light_mask.png", light_mask)
    # cv2.imshow("02 - img_bright_pass", img_bright_pass)
    # cv2.imwrite("02 - img_bright_pass.png", img_bright_pass * 255)
    # cv2.imshow("03 - blurred_bright_pass", blurred_bright_pass)
    # cv2.imwrite("03 - blurred_bright_pass.png", blurred_bright_pass * 255)
    cv2.imshow("04 - img_original", img)
    cv2.imwrite("04 - img_original.png", img * 255)
    cv2.imshow("05 - img_out", img_out)
    cv2.imwrite("05 - img_out.png", img_out * 255)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main ()

#===============================================================================