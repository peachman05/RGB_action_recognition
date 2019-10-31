import cv2

path_dataset = 'F:\\Master Project\\Dataset\\KARD-split-frames'
path_file = '\\a01\\a01_s01_e01\\frame0.jpg'
path = path_dataset+path_file
print(path)
image = cv2.imread(path,0)
# print(image)

new_image = cv2.resize(image,(200,200))
new_image = new_image/255.0
print(new_image.shape)
# print(new_image[0,0]/255.0)
cv2.imshow('image',new_image)
cv2.waitKey(0)