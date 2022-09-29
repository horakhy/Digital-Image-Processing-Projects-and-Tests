import sys
import timeit
import numpy as np
import cv2
import matplotlib.pyplot as plt

NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 5
LARGURA_MIN = 5
N_PIXELS_MIN = 5

INPUT_IMAGE = '82.bmp'

## flood fill and count the number of pixels
def flood_fill(image, x, y, color):
  if image[y, x] == 0:
    image[y, x] = color
    flood_fill(image, x + 1, y, color)
    flood_fill(image, x - 1, y, color)
    flood_fill(image, x, y + 1, color)
    flood_fill(image, x, y - 1, color)
    

def main():

  image = cv2.imread(INPUT_IMAGE) 
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  edged = cv2.Canny(gray, 80, 200) 
  
  # edged = cv2.GaussianBlur(edged, (3, 3), 0)
  
  contours, hierarchy = cv2.findContours(edged,  
      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
  amount = 0
  
  ## draw all contours with a minimum area of 60 pixels
  ## and a minimum width of 10 pixels
  for c in contours:
    if cv2.contourArea(c) > N_PIXELS_MIN:
      x, y, w, h = cv2.boundingRect(c)
      if w > LARGURA_MIN and h > ALTURA_MIN:
        amount += 1
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

  cv2.imshow('Canny Edges After Contouring', edged) 

  print("Number of Contours found = " + str(amount)) 
  cv2.drawContours(image, contours, -1, (0, 255, 0), 2) 

  cv2.imshow('Contours', image) 
  cv2.waitKey(0) 
  cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()
