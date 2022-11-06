import cv2

filepath = "dataset/256 copy 2.jpg"
image = cv2.imread(filepath)
print(image.shape)
print(type(image))