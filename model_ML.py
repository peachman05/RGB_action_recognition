from keras import Sequential
from keras.layers import CuDNNLSTM, LSTM
from keras.layers import Dense, Input, TimeDistributed, Conv2D
from keras.layers import Dropout, concatenate, Flatten, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.python.keras import optimizers
from keras.applications import MobileNet, MobileNetV2, ResNet152V2

from keras.regularizers import l2
from keras.models import Model


def create_model(dim, n_sequence, n_channels, n_output):
    model = Sequential()
    # after having Conv2D...
    model.add( 
        TimeDistributed(
            Conv2D(64, (3,3), activation='relu'), 
            input_shape=(n_sequence, *dim, n_channels) # 5 images...
        )
    )
    model.add(
        TimeDistributed(
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)) # Or Flatten()
        )
    )
    ###
    model.add(
        TimeDistributed(
            Conv2D(64, (3,3), activation='relu')
        )
    )
    # We need to have only one dimension per output
    # to insert them to the LSTM layer - Flatten or use Pooling
    model.add(
        TimeDistributed(
            GlobalAveragePooling2D() # Or Flatten()
        )
    )
    # previous layer gives 5 outputs, Keras will make the job
    # to configure LSTM inputs shape (5, ...)
    model.add(
        CuDNNLSTM(240, return_sequences=False)
    )
    # and then, common Dense layers... Dropout...
    # up to you
    model.add(Dense(240, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.5))
    # For example, for 3 outputs classes 
    model.add(Dense(n_output, activation='softmax'))
    sgd = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.01, nesterov=False)
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def create_model_pretrain(dim, n_sequence, n_channels, n_output, pretrain_name):
    model = Sequential()
    # after having Conv2D...
    if pretrain_name == 'ResNet152V2':
        model.add( 
            TimeDistributed(
                ResNet152V2(weights='imagenet',include_top=False), 
                input_shape=(n_sequence, *dim, n_channels) # 5 images...
            )
        )
    elif pretrain_name == 'MobileNet':
        model.add( 
            TimeDistributed(
                MobileNet(weights='imagenet',include_top=False), 
                input_shape=(n_sequence, *dim, n_channels) # 5 images...
            )
        )
    elif pretrain_name == 'MobileNetV2':
        model.add( 
            TimeDistributed(
                MobileNetV2(weights='imagenet',include_top=False), 
                input_shape=(n_sequence, *dim, n_channels) # 5 images...
            )
        )
    else:
        raise ValueError('pretrain_name is incorrect')

    model.add(
        TimeDistributed(
            GlobalAveragePooling2D() # Or Flatten()
        )
    )
    model.add(
        CuDNNLSTM(64, return_sequences=False)
    )
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.5))

    model.add(Dense(24, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(n_output, activation='softmax'))
    sgd = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.01, nesterov=False)
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_2stream_model(dim, n_sequence, n_channels, n_joint, n_output):

    rgb_stream = Input(shape=(n_sequence, *dim, n_channels), name='rgb_stream')     
    skeleton_stream = Input(shape=(n_sequence, n_joint), name='skleton_stream')

    mobileNet = TimeDistributed(MobileNetV2(weights='imagenet',include_top=False)) (rgb_stream)    
    rgb_feature = TimeDistributed(GlobalAveragePooling2D()) (mobileNet)

    rgb_lstm = CuDNNLSTM(64, return_sequences=False)(rgb_feature)
    skeleton_lstm = CuDNNLSTM(64, return_sequences=False)(skeleton_stream)

    combine = concatenate([rgb_lstm, skeleton_lstm])

    fc_1 = Dense(units=64, activation='relu')(combine)
    fc_1 = Dropout(0.5)(fc_1)
    fc_2 = Dense(units=24, activation='relu')(fc_1)
    fc_3 = Dense(units=4, activation='softmax', use_bias=True, name='main_output')(fc_2)
    model = Model(inputs=[rgb_stream,skeleton_stream], outputs=fc_3)
    sgd = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.01, nesterov=False)
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model