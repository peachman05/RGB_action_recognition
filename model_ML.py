from keras import Sequential
from keras.layers import CuDNNLSTM, LSTM
from keras.layers import Dense, Input, TimeDistributed, Conv2D, Conv3D, MaxPooling3D
from keras.layers import Dropout, concatenate, Flatten, GlobalAveragePooling2D, MaxPooling2D
from keras import optimizers
from keras.applications import MobileNet, MobileNetV2, ResNet152V2, Xception

from keras.regularizers import l2
from keras.models import Model
from keras import backend as K
import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)

def my_loss( y_true, y_pred ):
    y_pred_softmax  = K.softmax(y_pred) 
    return K.sparse_categorical_crossentropy(y_true, y_pred_softmax)

def mean_pred(y_true, y_pred):
    print("----------------")
    y_true = K.print_tensor(y_true, message='y_true = ')
    y_pred = K.print_tensor(y_pred, message='y_pred = ')
    y_pred_idx = K.argmax(y_pred, axis=-1)
    y_pred_idx = K.cast( y_pred_idx, K.dtype(y_true) )
    print('-------------',K.dtype(y_pred_idx))
    compare = K.equal(y_true, y_pred_idx)
    compare = K.print_tensor(compare, message='compare = ')
    return compare

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

def create_model_pretrain(dim, n_sequence, n_channels, n_output, alpha):
    model = Sequential()
    # after having Conv2D...
    # if pretrain_name == 'ResNet152V2':
    #     model.add( 
    #         TimeDistributed(
    #             ResNet152V2(weights='imagenet',include_top=False), 
    #             input_shape=(n_sequence, *dim, n_channels) # 5 images...
    #         )
    #     )
    # elif pretrain_name == 'Xception':
    #     model.add( 
    #         TimeDistributed(
    #             Xception(weights='imagenet',include_top=False), 
    #             input_shape=(n_sequence, *dim, n_channels) # 5 images...
    #         )
    #     )
    # elif pretrain_name == 'MobileNetV2':
    #     model.add( 
    #         TimeDistributed(
    #             # MobileNetV2(weights='imagenet',include_top=False), 
    #             MobileNetV2(weights='imagenet',include_top=False, alpha= alpha),
    #             input_shape=(n_sequence, *dim, n_channels) # 5 images...
    #         )
    #     )
    # else:
    #     raise ValueError('pretrain_name is incorrect')
    model.add( 
            TimeDistributed(
                MobileNetV2(weights='imagenet',include_top=False, alpha= alpha),
                input_shape=(n_sequence, *dim, n_channels) # 5 images...
            )
        )
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
    # model.add(Dense(n_output))
    model.add(Dense(n_output, activation='softmax'))

    # model.compile(optimizer='sgd', loss=my_loss, metrics=['sparse_categorical_accuracy'])
    
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_model_Conv3D(dim, n_sequence, n_channels, n_output, set_pretrain=False):
    model = Sequential()
    n_first_filter = 16
    model.add(  Conv3D(n_first_filter, kernel_size=(3, 3, 3), activation='relu',
             kernel_initializer='he_uniform',
             input_shape=(n_sequence,dim[0],dim[1],n_channels))
            )    
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(8, kernel_size=(3, 3, 3), activation='relu',
            kernel_initializer='he_uniform')
            )            
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(.4))
    model.add(Dense(24, activation='relu'))
    model.add(Dropout(.4))
    model.add(Dense(n_output, activation='softmax'))
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    if set_pretrain:
        mobile_model = Sequential()
        mobile_model.add(MobileNetV2(weights='imagenet',include_top=False))

        weights = model.layers[0].get_weights() # first layer
        weight_mobile = mobile_model.layers[0].get_weights()[0] # first layer, first weight

        for i in range(3):
            weights[0][:,:,i,:,:] = weight_mobile[:,:,:,:n_first_filter]

        model.layers[0].set_weights(weights)

    return model

def create_model_skeleton(n_sequence, n_joint, n_output):
    skeleton_stream = Input(shape=(n_sequence, n_joint*3 ), name='skleton_stream')
    skeleton_lstm = CuDNNLSTM(50, return_sequences=False)(skeleton_stream)
    skeleton_lstm = Dropout(0.5)(skeleton_lstm)
    fc_1 = Dense(units=60, activation='relu')(skeleton_lstm)
    fc_1 = Dropout(0.5)(fc_1)
    fc_2 = Dense(units=n_output, activation='softmax', use_bias=True, name='main_output')(fc_1)
    model = Model(inputs=skeleton_stream, outputs=fc_2)
    # print(model.summary())

    # model = Sequential()
    # model.add(CuDNNLSTM(64, input_shape=(n_sequence, n_joint*3 ),return_sequences=False))
    # model.add(Dropout(0.4))#使用Dropout函数可以使模型有更多的机会学习到多种独立的表征
    # model.add(Dense(24) )
    # model.add(Dropout(0.4))
    # model.add(Dense(n_output, activation='softmax'))
    # print(model.summary())

    sgd = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.01, nesterov=False)
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_2stream_model(dim, n_sequence, n_channels, n_joint, n_output):

    rgb_stream = Input(shape=(n_sequence, *dim, n_channels), name='rgb_stream')     
    skeleton_stream = Input(shape=(n_sequence, n_joint*3 ), name='skleton_stream')

    mobileNet = TimeDistributed(MobileNetV2(weights='imagenet',include_top=False)) (rgb_stream)    
    rgb_feature = TimeDistributed(GlobalAveragePooling2D()) (mobileNet)

    rgb_lstm = CuDNNLSTM(64, return_sequences=False)(rgb_feature)
    skeleton_lstm = CuDNNLSTM(64, return_sequences=False)(skeleton_stream)

    combine = concatenate([rgb_lstm, skeleton_lstm])

    fc_1 = Dense(units=64, activation='relu')(combine)
    fc_1 = Dropout(0.5)(fc_1)
    fc_2 = Dense(units=24, activation='relu')(fc_1)
    fc_2 = Dropout(0.3)(fc_2)
    fc_3 = Dense(units=n_output, activation='softmax', use_bias=True, name='main_output')(fc_2)
    model = Model(inputs=[rgb_stream,skeleton_stream], outputs=fc_3)
    sgd = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.01, nesterov=False)
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model