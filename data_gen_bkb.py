import numpy as np
import keras
import cv2
import os
import matplotlib.pyplot as plt

class DataGeneratorBKB(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32), n_channels=1,
                 n_classes=10, n_sequence=4, shuffle=True, path_dataset=None, type_gen='train'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_sequence = n_sequence
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.path_dataset = path_dataset
        self.type_gen = type_gen
        print("all:", len(self.list_IDs), " batch per epoch", int(np.floor(len(self.list_IDs) / self.batch_size)) )
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'        
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        if self.type_gen == 'predict':
            return X
        else:
            return X, y

    def get_sampling_frame(self, len_frames):   
        '''
        Sampling n_sequence frame from video file
        Output: n_sequence index from sampling algorithm 
        '''     
        
        random_sample_range = 4
        # Randomly choose sample interval and start frame
        sample_interval = np.random.randint(1, random_sample_range + 1)
        start_i = np.random.randint(0, len_frames - sample_interval * self.n_sequence + 1)
        
        # Extract frames as tensors
        index_sampling = []
        end_i = sample_interval * self.n_sequence + start_i
        for i in range(start_i, end_i, sample_interval):
            if len(index_sampling) < self.n_sequence:
                index_sampling.append(i)
        
        return index_sampling


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'         
        # print("ddddd")
        # Initialization
        X = np.empty((self.batch_size, self.n_sequence, *self.dim, self.n_channels)) # X : (n_samples, *dim, n_channels)
        Y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):  # ID is name of file
            path_file = self.path_dataset + ID
            cap = cv2.VideoCapture(path_file)
            length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get length of frames
            index_sampling = self.get_sampling_frame(length_file) # get index to sampling         
            for j, n_pic in enumerate(index_sampling):
                cap.set(cv2.CAP_PROP_POS_FRAMES, n_pic)
                ret, frame = cap.read()
                new_image = cv2.resize(frame, self.dim)
                new_image = new_image/255.0                
                X[i,j,:,:,:] = new_image

            Y[i] = self.labels[ID]
            cap.release()

        return X,Y

