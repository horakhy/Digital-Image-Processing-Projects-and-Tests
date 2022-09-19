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
THRESHOLD = 0.6
UPPER_LIMIT = 2
SIGMA = 7
KERNEL = 7

def gaussianBlur(img):
    sigma = 1
    for sigma in range(SIGMA):
        img += cv2.GaussianBlur(img, (KERNEL,KERNEL), sigma)
        sigma*=2
    
    return img

def boxBlur(img):
    for x in range(SIGMA):
        img += cv2.blur(img, (KERNEL,KERNEL))
        x+=1
    return img
#===============================================================================

def main ():

    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()
    img = img.reshape ((img.shape [0], img.shape [1], 3))

    # # Apenas o canal de luminosidade
    lightness = img[:,:,1]
    # # Criando imagem da máscara contendo as fontes de luz
    light_mask = cv2.inRange(lightness, 127, 255)

    start_time = timeit.default_timer()
    
    img = cv2.bitwise_and(img, img, mask = light_mask)
    img_out = boxBlur(img)
    print("Tempo: %f" % (timeit.default_timer() - start_time))

    cv2.imshow("02 - out", img_out)
    cv2.imwrite("02 - out.png", img_out * 255)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main ()

#===============================================================================