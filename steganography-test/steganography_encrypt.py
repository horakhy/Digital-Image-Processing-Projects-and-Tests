# Python program to demonstrate
# image steganography using OpenCV


import cv2

path = './composed/'

# Encryption function
def encrypt():
	
	# img1 and img2 are the
	# two input images
	img1 = cv2.imread('pic1.jpg')
	img2 = cv2.imread('pic2.jpg')
	
	for i in range(img2.shape[0]):
		for j in range(img2.shape[1]):
			for l in range(3):
				
				# v1 and v2 are 8-bit pixel values
				# of img1 and img2 respectively
				v1 = format(img1[i][j][l], '08b')
				v2 = format(img2[i][j][l], '08b')
				
				# Taking 4 MSBs of each image
				v3 = v1[:4] + v2[:4]
				
				img1[i][j][l]= int(v3, 2)
				
	cv2.imwrite(path + 'pic3-composed.png', img1)
	
# Driver's code
encrypt()
