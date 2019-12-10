from vidaug import augmentors as va
from matplotlib.pyplot import imread
import cv2
import numpy as np

sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
seq = va.Sequential([
    va.RandomCrop(size=(300, 300)), # randomly crop video with a size of (240 x 180)
    # va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
    # va.RandomTranslate(x=100,y=100), 
    va.Add(value=100),
    sometimes(va.HorizontalFlip()) # horizontally flip the video with 50% probability
])

image = cv2.imread("room.jpg")
image2 = cv2.imread("room3.jpg")
print(image.shape)
# Creating a dataset which contains just one image.
images = np.zeros((2, image.shape[0], image.shape[1], image.shape[2]))
images[0,:,:,:] = image
images[1,:,:,:] = image2

for batch_idx in range(2):
    # 'video' should be either a list of images from type of numpy array or PIL images
    # video = load_batch(batch_idx)
    video_aug = seq(images)
    # train_on_video(video)

    for i in range(2):
        cv2.imshow('Frame',video_aug[i]/255)
        if cv2.waitKey(2000) & 0xFF == ord('q'):
                break 


