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
import collections

#===============================================================================

INPUT_IMAGE =  'chave.bmp'
TAMANHO_JANELA = -40, 40



#===============================================================================

def esta_dentro_da_imagem (tamanho_imagem, y, x):
    return y >= 0 and y < tamanho_imagem[0] and x >= 0 and x < tamanho_imagem[1]

def calculo_janela_ingenuo (img, y, x, z):
    soma = 0
    num_pixels_janela = 0
    for janela_y in range(TAMANHO_JANELA[0], TAMANHO_JANELA[1]):
        for janela_x in range(TAMANHO_JANELA[0], TAMANHO_JANELA[1]):
            if esta_dentro_da_imagem(img.shape, y + janela_y, x + janela_x):
                soma += img [y + janela_y, x + janela_x, z]
                num_pixels_janela += 1
    return soma / num_pixels_janela


def blur_ingenuo (img):
    img_out = np.zeros_like (img)
    size_y = img.shape[0]
    size_x = img.shape[1]
    channels = img.shape[2]

    for z in range (channels):
        for y in range (size_y):
            for x in range (size_x):
                img_out [y, x, z] = calculo_janela_ingenuo (img, y, x, z)
    
    return img_out

def calculo_janela_separavel_y (img, y, x, z):
    soma = 0
    num_pixels_janela = 0
    for janela_y in range(TAMANHO_JANELA[0], TAMANHO_JANELA[1]):
        if esta_dentro_da_imagem(img.shape, y + janela_y, x):
            soma += img [y + janela_y, x, z]
            num_pixels_janela += 1
    return soma / num_pixels_janela

def calculo_janela_separavel_x (img, y, x, z):
    soma = 0
    num_pixels_janela = 0
    for janela_x in range(TAMANHO_JANELA[0], TAMANHO_JANELA[1]):
        if esta_dentro_da_imagem(img.shape, y, x + janela_x):
            soma += img [y, x + janela_x, z]
            num_pixels_janela += 1
    return soma / num_pixels_janela

def blur_separavel (img):
    img_out = np.zeros_like (img)
    size_y = img.shape[0]
    size_x = img.shape[1]
    channels = img.shape[2]

    for z in range (channels):
        for y in range (size_y):
            for x in range (size_x):
                img_out [y, x, z] = calculo_janela_separavel_y (img, y, x, z)

    img_out_final = np.zeros_like (img_out)
    for z in range (channels):
        for y in range (size_y):
            for x in range (size_x):
                img_out_final[y, x, z] = calculo_janela_separavel_x (img_out, y, x, z)
    
    return img_out_final

def cria_integral_da_imagem (img):
    img_integral = np.zeros_like (img)
    size_y = img.shape[0]
    size_x = img.shape[1]
    channels = img.shape[2]
    
    for z in range (channels):
        for y in range (size_y):
            img_integral[y,0, z] = img[y,0, z]
            for x in range (1, size_x):
                img_integral[y,x, z] = img[y,x, z] + img_integral[y,x-1, z]
          
        for y in range (1, size_y):
            for x in range (size_x):
                img_integral[y,x, z] += img_integral[y-1,x, z]
    
    return img_integral

def blur_imagem_integral (img):
    img_out = np.zeros_like (img)
    size_y = img.shape[0]
    size_x = img.shape[1]
    channels = img.shape[2]

    img_integral = cria_integral_da_imagem(img)

    for z in range (channels):
        janela = TAMANHO_JANELA[1]
        for y in range (int(size_y)):
            sum=0
            for x in range (int(size_x)):
                if(TAMANHO_JANELA[1] > x or TAMANHO_JANELA[1] > y):
                    janela = x if x < y else y
                    if((not(esta_dentro_da_imagem(img.shape, y+janela, x))) and y+janela > x+janela):
                        sum+=1
                        janela-=sum
                    elif((not(esta_dentro_da_imagem(img.shape, y, x+janela))) and x+janela > y+janela):
                        sum+=1
                        janela-=sum
                else:
                    if((not(esta_dentro_da_imagem(img.shape, y+janela, x))) and y+janela > x+janela):
                        janela-=1
                    elif((not(esta_dentro_da_imagem(img.shape, y, x+janela))) and x+janela > y+janela):
                        janela-=1
                if(janela!=0): 
                    img_out[y][x][z] = (img_integral[y+janela][x+janela][z] - img_integral[y-janela][x+janela][z] - img_integral[y+janela][x-janela][z] + img_integral[y-janela][x-janela][z])/(janela*janela*4)
                else:
                    img_out[y][x][z] = img[y][x][z]

    return img_out

#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 3))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    # img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.

    # cv2.imshow ('01 - original', img)
    # cv2.imwrite ('01 - original.png', img*255)

    start_time = timeit.default_timer ()
    img_out = blur_imagem_integral (img)
    img_out2 = cv2.blur (img, (80, 80))
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
