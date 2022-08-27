# Python program to demonstrate
# image steganography using OpenCV


import cv2
import numpy as np
import random

path = './decrypted/'

# Decryption function
def decrypt():
	
	# Encrypted image
	img = cv2.imread('./composed/pic3-composed.png')
	width = img.shape[0]
	height = img.shape[1]
	
	# img1 and img2 are two blank images
	img1 = np.zeros((width, height, 3), np.uint8)
	img2 = np.zeros((width, height, 3), np.uint8)
	
	for i in range(width):
		for j in range(height):
			for l in range(3):
				v1 = format(img[i][j][l], '08b')
				v2 = v1[:4] + chr(random.randint(0, 1)+48) * 4
				v3 = v1[4:] + chr(random.randint(0, 1)+48) * 4
				
				# Appending data to img1 and img2
				img1[i][j][l]= int(v2, 2)
				img2[i][j][l]= int(v3, 2)
	
	# These are two images produced from
	# the encrypted image
	cv2.imwrite(path + 'pic2_from_decryption.png', img1)
	cv2.imwrite(path + 'pic3_from_decryption.png', img2)
	
	
# Driver's code
decrypt()
