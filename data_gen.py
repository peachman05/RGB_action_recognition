import numpy as np
import keras
import cv2
import os

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32), n_channels=1,
                 n_classes=10, n_sequence=4, shuffle=True, path_dataset=None, type_gen='train'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_sequece = n_sequence
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.path_dataset = path_dataset
        self.type_gen = type_gen
        print("all:", len(self.list_IDs), " batch per epoch", int(np.floor(len(self.list_IDs) / self.batch_size)) )
        self.on_epoch_end()
        self.count = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        # print(self.count, len(X))
        self.count += 1

        if self.type_gen == 'predict':
            return X
        else:
            return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # print("innnnnn")
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_sampling_frame(self, folder_path):        
        
        len_frames = len(list( os.listdir(folder_path)))# dir is your directory path
        # print(self.isTrain,len_frames)
        # Start index and sample interval for the test set
        start_i, sample_interval = 0, len_frames // self.n_sequece
        if self.type_gen == 'train':
            # Randomly choose sample interval and start frame
            sample_interval = np.random.randint(1, len_frames // self.n_sequece + 1)
            start_i = np.random.randint(0, len_frames - sample_interval * self.n_sequece + 1)
        
        # Extract frames as tensors
        index_sampling = []
        end_i = sample_interval * self.n_sequece + start_i
        for i in range(start_i, end_i, sample_interval):
            if len(index_sampling) < self.n_sequece:
                index_sampling.append(i)
        
        return index_sampling


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.n_sequece, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):  
            path_folder = self.path_dataset + ID
            index_sampling = self.get_sampling_frame(path_folder)          
            for j, n_pic in enumerate(index_sampling):
                path = path_folder+'\\frame{:}.jpg'.format(n_pic)

                if self.n_channels > 0:
                    image = cv2.imread(path)
                else:
                    image = cv2.imread(path,0)

                new_image = cv2.resize(image, self.dim)
                new_image = new_image/255.0                
                X[i,j,:,:,:] = new_image

            Y[i] = self.labels[ID]


        return X,Y