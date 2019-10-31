from data_helper import readfile_to_dict
import numpy as np
import os

# from keras import Sequential
# from keras.layers import CuDNNLSTM, LSTM
# # from tensorflow.python.keras.layers import CuDNNLSTM
# from keras.layers import Dense, Input, TimeDistributed, Conv2D
# from keras.layers import Dropout, concatenate, Flatten, GlobalAveragePooling2D
# from keras.regularizers import l2
# from keras.models import Model
# from tensorflow.python.keras import optimizers
# from keras.applications import MobileNet

# dim = (10,10)
# X = np.empty((2, 4, *dim, 1))

# t = np.random.randint(0, 100, size=(10,10))
# new_image = np.reshape(t, (*dim, 1) )
# X[0,0,:] = t
# print(new_image.shape)

# path_dataset = 'F:\\Master Project\\Dataset\\KARD-split-frames'
# path_file = '\\a01\\a01_s01_e01'

# dir = path_dataset+path_file

# list = os.listdir(dir) # dir is your directory path
# number_files = len(list)
# print(number_files)

# d1 = readfile_to_dict("trainlist.txt")
# d2 = readfile_to_dict("testlist.txt")
# print(len(d1))
# all = d1.copy()
# all.update(d2)

# print(list(d1.keys()))
# print(len(d2))
# print(len(all))

# indexes = np.arange(105)
# # n = np.random.shuffle(indexes)
# # print(indexes)
# batch_size = 10
# index = 10
# indexes2 = indexes[index*batch_size:(index+1)*batch_size]
# print(indexes2)

# X = np.random.randint(0, 100, size=(4, 224, 224, 3)) # 4=batch size, (8,8,3) image
# Y = np.array([0, 0, 1, 2])
# print(X.shape)
# print(Y)

X = np.random.randint(0, 100, size=(4, 5, 224, 224, 3)) # 4=batch size, 5 sequence, (8,8,3) image
Y = np.array([0, 0, 1, 2])
print(X.shape)
print(Y)

# model = Sequential()
# model.add(MobileNet(weights='imagenet',include_top=False))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(4, activation='softmax'))
# sgd = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.01, nesterov=False)
# model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# dim = (224,224)
# n_channels = 3
# n_sequence = 5
# n_output = 4

# model = Sequential()
# # after having Conv2D...
# model.add( 
#     TimeDistributed(
#         MobileNet(weights='imagenet',include_top=False), 
#         input_shape=(n_sequence, *dim, n_channels) # 5 images...
#     )
# )
# model.add(
#     TimeDistributed(
#         GlobalAveragePooling2D() # Or Flatten()
#     )
# )
# # previous layer gives 5 outputs, Keras will make the job
# # to configure LSTM inputs shape (5, ...)
# model.add(
#     CuDNNLSTM(64, return_sequences=False)
# )
# # and then, common Dense layers... Dropout...
# # up to you
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(.5))

# model.add(Dense(24, activation='relu'))
# model.add(Dropout(.5))
# # For example, for 3 outputs classes 
# model.add(Dense(n_output, activation='softmax'))
# sgd = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.01, nesterov=False)
# model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# print(model.summary())
# model.fit(X,Y)
# dim = (224,224)
# n_sequence = 20
# n_channels = 3
# frame_window = np.empty((0, *dim, n_channels))
# print(frame_window.shape)

# new_f = np.random.randint(0, 100, size=(224, 224, 3))
# new_f = np.reshape(new_f, (1, *new_f.shape))
# print(new_f.shape)
# frame_window = np.append(frame_window, new_f, axis=0 )
# print(frame_window.shape)
# x = np.array([0]*25)
frame_window = np.empty((0, 3, 25))
new_f = np.random.randint(0, 100, size=(3, 25))
new_f = np.reshape(new_f, (1, *new_f.shape))
frame_window = np.append(frame_window, new_f, axis=0 )
print(frame_window.shape)
