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
THRESHOLD = 0.85

#===============================================================================

def main ():

    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    img = img.reshape((img.shape[0], img.shape[1], 3))
    img = img.astype(np.float32) / 255
    
    
    start_time = timeit.default_timer()
    # Bright Pass (?)
    img_out = np.where(img > THRESHOLD, img, 0.0)
    print("Tempo: %f" % (timeit.default_timer() - start_time))

    cv2.imshow("02 - out", img_out)
    cv2.imwrite("02 - out.png", img_out * 255)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main ()

#===============================================================================