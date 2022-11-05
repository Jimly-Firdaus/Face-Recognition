import cv2

filepath = "dataset/5px.png"
image = cv2.imread(filepath)
print(image.shape)