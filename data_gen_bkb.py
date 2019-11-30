import numpy as np
import keras
import cv2
import os
import matplotlib.pyplot as plt

class DataGeneratorBKB(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32), n_channels=1,
                 n_sequence=4, shuffle=True, path_dataset=None,
                 select_joint=[], type_gen='train', type_model='rgb'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_sequence = n_sequence
        self.shuffle = shuffle
        self.path_dataset = path_dataset
        self.select_joint = select_joint
        self.n_joint = len(select_joint)
        self.type_gen = type_gen
        self.type_model = type_model
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
        Input: 
            len_frames -- number of frames that this video have
        Output: 
            index_sampling -- n_sequence frame indexs from sampling algorithm 
        '''             
        # Define maximum sampling rate
        random_sample_range = 4
        if random_sample_range*self.n_sequence > len_frames:
            random_sample_range = len_frames//self.n_sequence
        
        # Randomly choose sample interval and start frame
        sample_interval = np.random.randint(1, random_sample_range + 1)
        
        # temp = len_frames - sample_interval * self.n_sequence + 1
        # if temp <= 0:
        #     print(temp, len_frames)
        start_i = np.random.randint(0, len_frames - sample_interval * self.n_sequence + 1)
        
        # Get n_sequence index of frames
        index_sampling = []
        end_i = sample_interval * self.n_sequence + start_i
        for i in range(start_i, end_i, sample_interval):
            if len(index_sampling) < self.n_sequence:
                index_sampling.append(i)
        
        return index_sampling


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X1 = np.empty((self.batch_size, self.n_sequence, *self.dim, self.n_channels)) # X : (n_samples, timestep, *dim, n_channels)
        X2 = np.empty((self.batch_size, self.n_sequence, self.n_joint*3))
        Y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):  # ID is name of file (2 batch)
            path_video = self.path_dataset + ID + '.mp4'
            path_skeleton = self.path_dataset + ID + '.npy'
            
            
            if self.type_model == '2stream' or self.type_model == 'rgb':
                cap = cv2.VideoCapture(path_video)    
                length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get how many frames this video have           
                
            if self.type_model == '2stream' or self.type_model == 'skeleton':
                skeleton_data = np.load(path_skeleton)
                length_file = skeleton_data.shape[0]

            index_sampling = self.get_sampling_frame(length_file) # get sampling index  
                
            if self.type_model == '2stream' or self.type_model == 'rgb':                
                # Get RGB sequence
                for j, n_pic in enumerate(index_sampling):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, n_pic) # jump to that index
                    ret, frame = cap.read()
                    new_image = cv2.resize(frame, self.dim)
                    new_image = new_image/255.0                
                    X1[i,j,:,:,:] = new_image            

            if self.type_model == '2stream' or self.type_model == 'skeleton':
                # Get skeleton sequence                   
                skeleton_data = skeleton_data[index_sampling]
                skeleton_data = skeleton_data[:,:,self.select_joint]
                skeleton_data = skeleton_data.reshape(self.n_sequence,self.n_joint*3)
                X2[i] = skeleton_data

            # Get label
            Y[i] = self.labels[ID]
            if self.type_model == '2stream' or self.type_model == 'rgb':
                cap.release()        

        if self.type_model == 'rgb':
            X = X1
        elif self.type_model == 'skeleton':
            X = X2
        elif self.type_model == '2stream':
            X = [X1, X2]

        return X,Y


# class DataGenerator2Stream(DataGeneratorBKB):
#     def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32),
#                  n_channels=1, n_sequence=4, shuffle=True, path_dataset=None, 
#                  select_joint=[], type_gen='train'):

#         super().__init__(list_IDs, labels, batch_size, dim, n_channels,
#                  n_sequence, shuffle, path_dataset, type_gen)
#         self.select_joint = select_joint
