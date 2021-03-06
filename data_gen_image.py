import numpy as np
import keras
import cv2
import os
# from vidaug import augmentors as va
from matplotlib.pyplot import imread, imshow, show
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_helper import calculateRGBdiff
# from imageai.Detection import ObjectDetection

class DataGeneratorBKB(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32), n_channels=1,
                 n_sequence=4, shuffle=True, path_dataset=None,
                 select_joint=[], type_gen='train', type_model='rgb', option=None):
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
        self.option = option
        self.type_gen = type_gen
        self.type_model = type_model
        print("all:", len(self.list_IDs), " batch per epoch", int(np.floor(len(self.list_IDs) / self.batch_size)) )
        
        # execution_path = os.getcwd()
        # self.detector = ObjectDetection()
        # self.detector.setModelTypeAsYOLOv3()
        # self.detector.setModelPath( os.path.join(execution_path , "pretrain/yolo.h5"))
        # self.detector.loadModel(detection_speed="fast")#detection_speed="fast"
        # self.execution_path = execution_path
        # self.detector = detector

        # sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
        # self.seq = va.SomeOf([ #va.Sequential([
        #     # va.RandomCrop(size=(300, 300)), # randomly crop video with a size of (240 x 180)
        #     va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
        #     va.RandomTranslate(x=60,y=30), 
        #     # sometimes(va.Add(value=-100)),
        #     # sometimes(va.Pepper(ratio=40)),
        #     sometimes(va.Add(value=-60)),
        #     sometimes(va.HorizontalFlip()) # horizontally flip the video with 50% probability
        # ], 2)

        self.aug_gen = ImageDataGenerator()        
        
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

    def get_sampling_frame(self, len_frames, path_video):   
        '''
        Sampling n_sequence frame from video file
        Input: 
            len_frames -- number of frames that this video have
        Output: 
            index_sampling -- n_sequence frame indexs from sampling algorithm 
        '''

        # Define maximum sampling rate
        # sample_interval = len_frames//self.n_sequence
        # start_i = 0 #np.random.randint(0, len_frames - sample_interval * self.n_sequence + 1)

        if True:#self.type_gen =='train':
            random_sample_range = 10
            if random_sample_range*self.n_sequence > len_frames:
                random_sample_range = len_frames//self.n_sequence

            if random_sample_range <= 0:
                print('test:',random_sample_range, len_frames, path_video)
            # Randomly choose sample interval and start frame
            if random_sample_range < 3:
                sample_interval = np.random.randint(1, random_sample_range + 1)
            else:
                sample_interval = np.random.randint(3, random_sample_range + 1)

            # sample_interval = np.random.randint(1, random_sample_range + 1)

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

    def get_crop_img(self, frame):
        # detect_image, detections, extract_picture = self.detector.detectObjectsFromImage(input_type="array", input_image=frame, output_type='array', 
        #                                          minimum_percentage_probability=10, extract_detected_objects=True )
        print('#################',self.execution_path)
        detections = self.detector.detectObjectsFromImage(input_image=os.path.join(self.execution_path , "room.jpg"),
             output_image_path=os.path.join(self.execution_path , "image2new.jpg"), minimum_percentage_probability=30)
        max_prob = 0
        max_idx = 0
        for i,eachObject in enumerate(detections):
            if eachObject["name"] == 'person' and eachObject["percentage_probability"] > max_prob:
                max_prob = eachObject["percentage_probability"]
                max_idx = i
        if max_idx > len(detections):
            # if no detection, use black array
            crop_img = np.zeros((*self.dim, self.n_channels))
        else:
            crop_img = extract_picture[max_idx]
        return crop_img

    # def calculateRGBdiff(self, sequence_img):
    #     'keep first frame as rgb data, other is use RGBdiff for temporal data'
    #     length = len(sequence_img)
    #     new_sequence = np.zeros((length,self.dim[0],self.dim[1],self.n_channels))

    #     # find RGBdiff frame 1 to last frame
    #     for i in range(length-1,3,-1): # count down
    #         new_sequence[i] = cv2.subtract(sequence_img[i],sequence_img[i-1])
        
    #     new_sequence[:4] = sequence_img[:4] # first frame as rgb data

    #     return new_sequence

    def sequence_augment(self, sequence):
        name_list = ['rotate','width_shift','height_shift',
                    'brightness','flip_horizontal','width_zoom',
                    'height_zoom']
        dictkey_list = ['theta','ty','tx',
                    'brightness','flip_horizontal','zy',
                    'zx']
        # dictkey_list = ['ty','tx','zy','zx']
        random_aug = np.random.randint(2, 5) # random 0-4 augmentation method
        pick_idx = np.random.choice(len(dictkey_list), random_aug, replace=False) #

        dict_input = {}
        for i in pick_idx:
            if dictkey_list[i] == 'theta':
                # dict_input['theta'] = np.random.randint(-10, 10)
                dict_input['theta'] = np.random.randint(-5,5)

            elif dictkey_list[i] == 'ty': # width_shift
                # dict_input['ty'] = np.random.randint(-60, 60)
                dict_input['ty'] = np.random.randint(-20,20)

            elif dictkey_list[i] == 'tx': # height_shift
                # dict_input['tx'] = np.random.randint(-30, 30)
                dict_input['tx'] = np.random.randint(-10,10)

            elif dictkey_list[i] == 'brightness': 
                dict_input['brightness'] = np.random.uniform(0.15,1)

            elif dictkey_list[i] == 'flip_horizontal': 
                dict_input['flip_horizontal'] = True

            elif dictkey_list[i] == 'zy': # width_zoom
                # dict_input['zy'] = np.random.uniform(0.5,1.5)
                dict_input['zy'] = np.random.uniform(0.9,1.3) 

            elif dictkey_list[i] == 'zx': # height_zoom
                # dict_input['zx'] = np.random.uniform(0.5,1.5)
                dict_input['zx'] = np.random.uniform(0.9,1.3) 
        
        sh = sequence.shape
        new_sequence = np.zeros((sh[0],sh[1],sh[2],sh[3]))
        for i in range(sh[0]):
            new_sequence[i] = self.aug_gen.apply_transform(sequence[i],dict_input)
        
        return new_sequence
        

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X1 = np.empty((self.batch_size, self.n_sequence, *self.dim, self.n_channels)) # X : (n_samples, timestep, *dim, n_channels)
        X2 = np.empty((self.batch_size, self.n_sequence, self.n_joint*3))
        Y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):  # ID is name of file (2 batch)
            path_video = self.path_dataset + ID + '.mp4'
            
            cap = cv2.VideoCapture(path_video)    
            length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get how many frames this video have ,-1 because some bug          
                
            index_sampling = self.get_sampling_frame(length_file, path_video) # get sampling index  
           
            # Get RGB sequence
            for j, n_pic in enumerate(index_sampling):
                cap.set(cv2.CAP_PROP_POS_FRAMES, n_pic) # jump to that index
                ret, frame = cap.read()
                # frame = self.get_crop_img(frame)
                new_image = cv2.resize(frame, self.dim)
                # new_image = frame
                # new_image = new_image/255.0                
                X1[i,j,:,:,:] = new_image
                

            if self.type_gen =='train':
                X1[i,] = self.sequence_augment(X1[i,])/255.0*2-1
            else:
                X1[i,] = X1[i,]/255.0*2-1

            if self.option == 'RGBdiff':
                # print("dddddddddddd")
                X1[i,] = calculateRGBdiff(X1[i,], 0)
                        
            # Get label
            Y[i] = self.labels[ID]
            cap.release()        

        X = X1

        return X,Y


# class DataGenerator2Stream(DataGeneratorBKB):
#     def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32),
#                  n_channels=1, n_sequence=4, shuffle=True, path_dataset=None, 
#                  select_joint=[], type_gen='train'):

#         super().__init__(list_IDs, labels, batch_size, dim, n_channels,
#                  n_sequence, shuffle, path_dataset, type_gen)
#         self.select_joint = select_joint
