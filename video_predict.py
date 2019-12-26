from data_helper import readfile_to_dict
import numpy as np
import os
import cv2
from model_ML import create_model_pretrain
# import matplotlib.pyplot as plt

dim = (224,224)
n_sequence = 8
n_channels = 3
n_output = 5
batch_size = 1

def get_sampling_frame( len_frames):   
    '''
    Sampling n_sequence frame from video file
    Output: n_sequence index from sampling algorithm 
    '''     
    
    random_sample_range = 1
    # Randomly choose sample interval and start frame
    sample_interval = 3#np.random.randint(1, random_sample_range + 1)
    # print('sample_interval:',sample_interval)
    start_i = 30 #np.random.randint(0, len_frames - sample_interval * n_sequence + 1)

    # sample_interval = len_frames//n_sequence
    # start_i = 0
    
    # Extract frames as tensors
    index_sampling = []
    end_i = sample_interval * n_sequence + start_i
    for i in range(start_i, end_i, sample_interval):
        if len(index_sampling) < n_sequence:
            index_sampling.append(i)
    
    return index_sampling


def calculateRGBdiff(sequence_img):
    'keep first frame as rgb data, other is use RGBdiff for temporal data'
    length = len(sequence_img)
    sh = sequence_img.shape
    new_sequence = np.zeros((sh[0],sh[1],sh[2],sh[3])) # (frame, w,h,3)

    # find RGBdiff frame 1 to last frame
    for i in range(length-1,0,-1): # count down
        new_sequence[i] = cv2.subtract(sequence_img[i],sequence_img[i-1])
    
    new_sequence[0] = sequence_img[0] # first frame as rgb data

    return new_sequence


X = np.empty((batch_size, n_sequence, *dim, n_channels)) # X : (n_samples, *dim, n_channels)
Y = np.empty((batch_size), dtype=int)

sub_folder = 'original'
action = 'walk'
base_path = 'F:\\Master Project\\'
# base_path = 'D:\\Peach\\'
# path_file = base_path+'Dataset\\sit_stand\\'+action+'\\'+action+'03_04.mp4'
# path_file = base_path+'Dataset\\KARD-split\\'+action+'\\'+action+'_s09_e03.mp4'
path_file = 'F:\\Master Project\\Dataset\\BUPT-dataset\\RGBdataset\\'+sub_folder+'\\'+action+'\\'+action+'01_01.mp4'
# path_file = 'F:\\Master Project\\Dataset\\BasketBall-RGB\\'+action+'\\'+action+'00.mp4'
cap = cv2.VideoCapture(path_file)
length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get length of frames,
print(length_file)
index_sampling = get_sampling_frame(length_file) # get index to sampling         
for j, n_pic in enumerate(index_sampling):
    print(j, n_pic)

    cap.set(cv2.CAP_PROP_POS_FRAMES, n_pic)
    ret, frame = cap.read()
    new_image = cv2.resize(frame, dim)
    new_image = new_image/255.0                
    X[0,j,:,:,:] = new_image
    
    # cv2.imshow('Frame',frame)
    # cv2.waitKey(500)

cap.release()
print(X.shape)

## Predict
weights_path = 'BUPT-augment-RGBdiff-120-0.90-0.91.hdf5'
# weights_path = 'KARD-aug-RGBdif-40-0.92-0.98.hdf5'
model = create_model_pretrain(dim, n_sequence, n_channels, n_output, 'MobileNetV2')
model.load_weights(weights_path)

X[0,] = calculateRGBdiff(X[0,])

for i in range(n_sequence):    
    cv2.imshow('Frame',X[0,i])
    cv2.waitKey(500)
result = model.predict(X)
print(result)
# class_label = ['run','sit','stand','standup','walk']