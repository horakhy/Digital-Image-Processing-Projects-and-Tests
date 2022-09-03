#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'chave.bmp'
TAMANHO_JANELA = -10, 10



#===============================================================================

def esta_dentro_da_imagem (tamanho_imagem, y, x):
    return y >= 0 and y < tamanho_imagem[0] and x >= 0 and x < tamanho_imagem[1]

def calculo_janela_ingenuo (img, y, x):
    soma = 0
    num_pixels_janela = 0
    for janela_y in range(TAMANHO_JANELA[0], TAMANHO_JANELA[1]):
        for janela_x in range(TAMANHO_JANELA[0], TAMANHO_JANELA[1]):
            if esta_dentro_da_imagem(img.shape, y + janela_y, x + janela_x):
                soma += img [y + janela_y, x + janela_x]
                num_pixels_janela += 1
    return soma / num_pixels_janela


def calculo_janela_separavel_y (img, y, x):
    soma = 0
    num_pixels_janela = 0
    for janela_y in range(TAMANHO_JANELA[0], TAMANHO_JANELA[1]):
        if esta_dentro_da_imagem(img.shape, y + janela_y, x):
            soma += img [y + janela_y, x]
            num_pixels_janela += 1
    return soma / num_pixels_janela

def calculo_janela_separavel_x (img, y, x):
    soma = 0
    num_pixels_janela = 0
    for janela_x in range(TAMANHO_JANELA[0], TAMANHO_JANELA[1]):
        if esta_dentro_da_imagem(img.shape, y, x + janela_x):
            soma += img [y, x + janela_x]
            num_pixels_janela += 1
    return soma / num_pixels_janela


def blur_ingenuo (img):
    img_out = np.zeros_like (img)
    size_y = img.shape[0]
    size_x = img.shape[1]

    for y in range (size_y):
        for x in range (size_x):
            img_out [y, x] = calculo_janela_ingenuo (img, y, x)
    return img_out

def blur_separavel (img):
    img_out = np.zeros_like (img)
    size_y = img.shape[0]
    size_x = img.shape[1]

    for y in range (size_y):
        for x in range (size_x):
            img_out [y, x] = calculo_janela_separavel_y (img, y, x)

    img_out_final = np.zeros_like (img_out)
    for y in range (size_y):
        for x in range (size_x):
            img_out_final[y, x] = calculo_janela_separavel_x (img_out, y, x)
    return img_out_final

#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.

    # cv2.imshow ('01 - original', img)
    # cv2.imwrite ('01 - original.png', img*255)

    start_time = timeit.default_timer ()
    img_out = blur_separavel (img)
    img_out2 = cv2.blur (img, (20, 20))
    print ('Tempo: %f' % (timeit.default_timer () - start_time))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.imshow ('03 - openCV-out', img_out2)
    cv2.imwrite ('03 - openCV-out.png', img_out2*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
