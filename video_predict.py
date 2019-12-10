from data_helper import readfile_to_dict
import numpy as np
import os
import cv2
from model_ML import create_model_pretrain
# import matplotlib.pyplot as plt

dim = (224,224)
n_sequence = 8
n_channels = 3
n_output = 3
batch_size = 1
def get_sampling_frame( len_frames):   
    '''
    Sampling n_sequence frame from video file
    Output: n_sequence index from sampling algorithm 
    '''     
    
    random_sample_range = 1
    # Randomly choose sample interval and start frame
    sample_interval = 1#np.random.randint(1, random_sample_range + 1)
    # print('sample_interval:',sample_interval)
    start_i = 60 #np.random.randint(0, len_frames - sample_interval * n_sequence + 1)

    # sample_interval = len_frames//n_sequence
    # start_i = 0
    
    # Extract frames as tensors
    index_sampling = []
    end_i = sample_interval * n_sequence + start_i
    for i in range(start_i, end_i, sample_interval):
        if len(index_sampling) < n_sequence:
            index_sampling.append(i)
    
    return index_sampling


X = np.empty((batch_size, n_sequence, *dim, n_channels)) # X : (n_samples, *dim, n_channels)
Y = np.empty((batch_size), dtype=int)


action = 'standup'
base_path = 'F:\\Master Project\\'
# base_path = 'D:\\Peach\\'
path_file = base_path+'Dataset\\sit_stand\\'+action+'\\'+action+'03_04.mp4'
# path_file = 'F:\\Master Project\\Dataset\\BUPT-dataset\\RGBdataset\\'+action+'\\'+action+'01_01.mp4'
# path_file = 'F:\\Master Project\\Dataset\\BasketBall-RGB\\'+action+'\\'+action+'00.mp4'
cap = cv2.VideoCapture(path_file)
length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get length of frames
print(length_file)
index_sampling = get_sampling_frame(length_file) # get index to sampling         
for j, n_pic in enumerate(index_sampling):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_pic)
    ret, frame = cap.read()
    new_image = cv2.resize(frame, dim)
    new_image = new_image/255.0                
    X[0,j,:,:,:] = new_image
    print(j)

    cv2.imshow('Frame',frame)
    cv2.waitKey(500)

cap.release()
print(X.shape)

## Predict
# weights_path = 'pretrain/BUPT-28-0.97-0.98.hdf5'
weights_path = 'Sit-augment-30-0.85-0.85.hdf5'
model = create_model_pretrain(dim, n_sequence, n_channels, n_output, 'MobileNetV2')
model.load_weights(weights_path)
result = model.predict(X)
print(result)
class_label = ['run','sit','stand','standup','walk']