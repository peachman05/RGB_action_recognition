from data_helper import readfile_to_dict
import numpy as np
import os
import cv2
from model_ML import create_model_pretrain
import matplotlib.pyplot as plt

# new_f = np.random.randint(0, 100, size=(224, 224, 3))
# new_f = np.reshape(new_f, (1, *new_f.shape))
# print(new_f.shape)
# frame_window = np.append(frame_window, new_f, axis=0 )
# print(frame_window.shape)
# x = np.array([0]*25)

# name = 'F:\\Master Project\\Dataset\\KARD-split\\a01\\a01_s01_e02.mp4'
# cap = cv2.VideoCapture(name)
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# print( length )
# cap.set(cv2.CAP_PROP_POS_FRAMES, 123)

# while True:
#     ret, frame = cap.read()
    
#     print(frame)
#     if ret == False:
#         break
#     if cv2.waitKey(20) & 0xFF == ord('q') :
#         break
#     cv2.imshow("Video", frame)

# cv2.destroyAllWindows()
# test = {'test':1, 'test':2}
# print(test)


dim = (224,224)
n_sequence = 8
n_channels = 3
n_output = 4
batch_size = 1
def get_sampling_frame( len_frames):   
    '''
    Sampling n_sequence frame from video file
    Output: n_sequence index from sampling algorithm 
    '''     
    
    random_sample_range = 4
    # Randomly choose sample interval and start frame
    sample_interval = 1 #np.random.randint(1, random_sample_range + 1)
    start_i = np.random.randint(0, len_frames - sample_interval * n_sequence + 1)
    
    # Extract frames as tensors
    index_sampling = []
    end_i = sample_interval * n_sequence + start_i
    for i in range(start_i, end_i, sample_interval):
        if len(index_sampling) < n_sequence:
            index_sampling.append(i)
    
    return index_sampling


X = np.empty((batch_size, n_sequence, *dim, n_channels)) # X : (n_samples, *dim, n_channels)
Y = np.empty((batch_size), dtype=int)

# for i, ID in enumerate(list_IDs_temp):  # ID is name of file

action = 'stand'
path_file = 'F:\\Master Project\\Dataset\\BasketBall-RGB\\'+action+'\\'+action+'00.mp4'
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
    # imgplot = plt.imshow(frame)
    # plt.show()
    cv2.waitKey(500)

# Y[i] = labels[ID]
cap.release()
# for i in range(15):
    #  = new_image
    # cv2.imshow('Frame',X[0,i,:,:,:])