from matplotlib.pyplot import imread, imshow, show
import numpy as np
from vidaug import augmentors as va
import cv2

# image = imread("room.jpg")
# image2 = imread("room3.jpg")

# imshow(image-image2)
# show()

action = 'sitdown'
base_path = 'F:\\Master Project\\'
# base_path = 'D:\\Peach\\'
path_file = base_path+'Dataset\\sit_stand\\'+action+'\\'+action+'00_06.mp4'

cap = cv2.VideoCapture(path_file)
length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get length of frames
print(length_file)
# index_sampling = get_sampling_frame(length_file) # get index to sampling   

# cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
# ret, frame1 = cap.read()
# ret, frame2 = cap.read()

# imshow(frame1)
# show()

# cv2.imshow('Frame',frame1)
# cv2.waitKey(2000)
# cv2.imshow('Frame',frame2)
# cv2.waitKey(2000)
# cv2.imshow('Frame',frame2-frame1)
# cv2.waitKey(10000)

print("dd")

sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
seq = va.Sequential([
    # va.RandomCrop(size=(300, 300)), # randomly crop video with a size of (240 x 180)
    va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
    va.RandomTranslate(x=60,y=30), 
    # va.Add(value=-100),
    # va.Pepper(ratio=40),
    # va.Add(value=100),
    va.HorizontalFlip() # horizontally flip the video with 50% probability
])

def calculateRGBdiff( sequence_img, dim):
    'keep first frame as rgb data, other is use RGBdiff for temporal data'
    length = len(sequence_img)
    new_sequence = np.zeros((length,dim[0],dim[1],3),dtype='uint8')

    # find RGBdiff frame 1 to last frame
    for i in range(length-1,0,-1): # count down
        new_sequence[i] = cv2.subtract(sequence_img[i],sequence_img[i-1])
    
    new_sequence[0] = sequence_img[0] # first frame as rgb data

    return new_sequence
# cv2.imshow('Frame',frame)
# cv2.waitKey(500)
ret, frame = cap.read()
# cv2.imshow('Frame', frame)
# cv2.waitKey(3000)
dim = frame.shape
print(dim)
sequence = np.empty((length_file,dim[0],dim[1],3),dtype='uint8') #
pre_frame = frame
for j in range(0,length_file-1):
    ret, frame = cap.read()    
    print(j)
    sequence[j] = frame
    # if j%2 == 0:
    #     dif = cv2.subtract(frame,pre_frame)
    # print(sequence[j,].shape)
    # cv2.imshow('Frame', sequence[j,] )
    # cv2.waitKey(100)
    #     pre_frame = frame

sequence_new = calculateRGBdiff(sequence,dim)
sequence_new = np.array(seq(sequence_new))
# sequence_new = sequence
for j in range(0,length_file):
    sequence_new[j] = sequence_new[j]/255.0
    cv2.imshow('Frame', sequence_new[j] )
    cv2.waitKey(50)
