from model_ML import create_model_pretrain#, create_model_Conv3D
from keras import Sequential
from keras.layers import CuDNNLSTM, LSTM
from keras.layers import Dense, Input, TimeDistributed, Conv2D, Conv3D, MaxPooling3D
from keras.layers import Dropout, concatenate, Flatten, GlobalAveragePooling2D, MaxPooling2D
from keras import optimizers
from keras.applications import MobileNet, MobileNetV2, ResNet152V2, Xception

def create_model_Conv3D(dim, n_sequence, n_channels, n_output):
    model = Sequential()
    model.add(  Conv3D(16, kernel_size=(3, 3, 3), activation='relu',
             kernel_initializer='he_uniform',
             input_shape=(n_sequence,dim[0],dim[1],n_channels))
            )    
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(8, kernel_size=(3, 3, 3), activation='relu',
            kernel_initializer='he_uniform')
            )            
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.4))
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(.4))
    model.add(Dense(n_output, activation='softmax'))
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

dim_1 = (120,120)
dim_2 = (120,120)
n_sequence_1 = 10
n_sequence_2 = 8
n_channels = 3
n_output = 5
model_1 = create_model_Conv3D(dim_1, n_sequence_1, n_channels, n_output) 
model_2 = create_model_pretrain(dim_2, n_sequence_2, n_channels, n_output, 1.0)
model_3 = create_model_pretrain(dim_2, n_sequence_2, n_channels, n_output, 0.35)

print(model_1.summary())
print(model_2.summary())
print(model_3.summary())