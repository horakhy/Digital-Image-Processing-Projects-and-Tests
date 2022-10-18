import cv2
import sys
import numpy as np

path = '60.bmp'

def processing(img):
    img_out = img.copy()
    kernel = np.ones((5, 5), np.uint8)

    img_out = cv2.GaussianBlur(img_out, (5, 5), 2)
    cv2.imwrite('00 - GaussianBlur.png', img_out)

    img_out = cv2.adaptiveThreshold(img_out, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 19, -21)
    cv2.imwrite('01 - Adaptative_Threshold.png', img_out)

    img_out = cv2.morphologyEx(img_out, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite('02 - Morphology.png', img_out)

    return img_out

image = cv2.imread(path, 0)
if image is None:
    print('Erro abrindo a imagem.\n')
    sys.exit()

processed = processing(image)

(cnts, hierarchy) = cv2.findContours(
    processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
totalArea = 0.0
listArea = []
print(' lista de contornos:', len(cnts))
for x in range(len(cnts)):
    listArea.append(cv2.contourArea(cnts[x]))
    totalArea += listArea[x]

listAreaSorted = sorted(listArea)
print('lista de areas:', listAreaSorted)
print('area total: ', totalArea)

finalList = []
somaMenores = listAreaSorted[0]
for x in range(len(listAreaSorted)):
    if (somaMenores < totalArea*(0.01)):
        somaMenores += listAreaSorted[x]
        continue
    finalList = listAreaSorted[x-1:]
    break
somaListaFinal = 0.0
for x in range(len(finalList)):
    somaListaFinal += finalList[x]

somaListaFinal = 0.0
for x in range(len(finalList)):

    somaListaFinal += finalList[x]
print('Soma final:', somaListaFinal)

print('lista de areas final:', finalList)

mean = (totalArea-(somaMenores)) / len(finalList)
print('media:', mean)

rices = 0.0
for x in range(len(finalList)):
    if (finalList[x] < mean*0.2):
        continue
    if (finalList[x] > mean):
        rices += round(finalList[x]/mean)
        continue
    rices += 1
if (rices*mean - somaListaFinal > 0):
    rices += (rices*mean - somaListaFinal)/mean

cv2.drawContours(rgb, cnts, -1, (0, 0, 255), 1)
cv2.imwrite('03 - Draw.png', rgb)
rices = round(rices)
print('Rice in the image: ', rices)
# cv2.show()
