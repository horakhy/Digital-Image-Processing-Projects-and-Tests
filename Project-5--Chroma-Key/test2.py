#===============================================================================
# Autores:
#   Heitor Tonel Ventura - 2086883
#   José Henrique Ivanchechen - 2090341
#===============================================================================

import os
import cv2
import numpy as np

# Extremos a serem truncados da másica de distâncias euclideanas
TRUNC_RANGE = (0.1, 0.4)
# Ângulo mínimo representando o alcance de verde no espaço HSV
LOWER_GREEN = 70
# Ângulo máximo representando o alcance de verde no espaço HSV
UPPER_GREEN = 140

# Imagens de entrada
IN = 'img/'
# Pasta onde serão salvas as imagens de saída
OUT = ''
# Imagem de fundo que substituirá o fundo verde na imagem de entrada
BG = 'Wind_Waker_GC.bmp'

# Entra uma imagem com entonação verde e retorna uma máscara dizendo o que é verde (fundo) de acordo com os parâmetros
def create_green_mask(img, s=0.5, v=0.5):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  lower_green = np.array([LOWER_GREEN, s, v]).astype(np.float32)
  upper_green = np.array([UPPER_GREEN, 1, 1]).astype(np.float32)

  mask = cv2.inRange(hsv, lower_green, upper_green).astype(np.float32) / 255
  return mask

# Retorna a cor média da imagem de entrada dos pixels na máscara informada
def determine_key(fg, bg_mask):
  bg_mask = np.dstack((bg_mask, bg_mask, bg_mask))
  bg = fg * bg_mask

  r = bg[:, :, 2]
  g = bg[:, :, 1]
  b = bg[:, :, 0]

  avg_r = sum(r[r > 0]) / r[r > 0].size
  avg_g = sum(g[g > 0]) / g[g > 0].size
  avg_b = sum(b[b > 0]) / b[b > 0].size
  return (avg_b, avg_g, avg_r)

# Reduz a quantidade de verde contida em regiões da imagem de entrada com base na máscara
def reduce_green(img, mask):
  out = img.copy()

  green_mask = create_green_mask(img, 0.05, 0.05)

  reds = out[:, :, 2]
  greens = out[:, :, 1]
  blues = out[:, :, 0]

  cond = np.logical_and(mask > 0, green_mask > 0)

  # Faz o canal verde ser balanceado para a média do canal azul e vermelho no RGB
  greens[cond] = (reds[cond] + blues[cond]) / 2

  return out

# Constrói uma máscara de escala de cinza baseada na distância euclidiana entre a cor média do fundo verde e as cores da imagem de entrada
def build_euclidean_distances_mask(fg):
  # Cria máscara do que é com muito provavelmente fundo
  bg_mask = create_green_mask(fg)

  # Calcula a cor média do verde da máscara do fundo
  key = determine_key(fg, bg_mask)

  # Converte cor chave de RGB para LAB
  key_lab = cv2.cvtColor(np.float32([[key]]), cv2.COLOR_BGR2LAB)[0][0]

  # Converte imagem de entrada de RGB para LAB
  lab = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB)

  # Calcula distância euclidiana entre a cor chave e cada pixel da imagem de entrada
  euclidean = np.sqrt((lab[:, :, 0] - key_lab[0]) * 2 + (lab[:, :, 1] - key_lab[1]) * 2 + (lab[:, :, 2] - key_lab[2]) ** 2)

  # Normaliza as distâncias euclidianas entre 0 e 1
  euclidean = cv2.normalize(euclidean, euclidean, 0, 1, cv2.NORM_MINMAX)

  # Trunca tudo fora do alcance determinado e normaliza novamente
  euclidean = np.clip(euclidean, *TRUNC_RANGE, euclidean)
  euclidean = cv2.normalize(euclidean, euclidean, 0, 1, cv2.NORM_MINMAX)

  return euclidean

# Interpola as imagens da frente e do fundo com base numa máscara em escala de cinza
def interpolate(fg, bg, mask):
  mask = np.dstack((mask, mask, mask))
  return (fg * mask) + bg * (1 - mask)

# Realiza toda a operação de chroma key, com base numa imagem de entrada e um fundo
def chroma_key(fg_img, bg_img):
  # 1. Calcula as distâncias entre a cor média do fundo verde e as cores da imagem de entrada
  # 2. Trunca extremos e normaliza as distâncias
  euclidean_mask = build_euclidean_distances_mask(fg_img)

  # Reduz a quantidade de verde contida na imagem de entrada com base na máscara de distâncias
  fg_img = reduce_green(fg_img, euclidean_mask)

  # Interpola a imagem de foreground com a de background com base na máscara de distâncias normalizadas
  final_img = interpolate(fg_img, bg_img, euclidean_mask)

  return final_img, euclidean_mask

def main():
  # Imagem de fundo
  bg_img = cv2.imread(BG)
  bg_img = bg_img.astype(np.float32) / 255

  for filename in os.listdir(IN):
    if filename.lower().endswith('.bmp'):
      print(filename, 'start')
      # Imagem de entrada
      fg_img = cv2.imread(IN + filename)
      fg_img = fg_img.astype(np.float32) / 255

      bg = cv2.resize(bg_img, (fg_img.shape[1], fg_img.shape[0]))

      final_img, mask = chroma_key(fg_img, bg)
      cv2.imwrite(OUT + filename, final_img * 255)
      cv2.imwrite(OUT + 'mask' + filename, mask * 255)
      print(filename, 'end')

if __name__ == '__main__':
  main()