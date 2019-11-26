from keras import Sequential
from keras.layers import CuDNNLSTM, LSTM
# from tensorflow.python.keras.layers import CuDNNLSTM
from keras.layers import Dense, Input, TimeDistributed, GlobalAveragePooling2D
from keras.layers import Dropout, concatenate, Flatten
from keras.regularizers import l2
from keras.models import Model
from keras.applications import MobileNetV2
from tensorflow.python.keras import optimizers


def create_2stream_model(num_frame, num_joint):

    up = Input(shape=(num_frame, num_joint), name='up_stream')
    down = Input(shape=(num_frame, num_joint), name='down_stream')

    up_feature = CuDNNLSTM(64, return_sequences=False)(up)
    down_feature = CuDNNLSTM(64, return_sequences=False)(down)
    # up_feature = Flatten()(up)
    # down_feature = Flatten()(down)

    feature = concatenate([up_feature, down_feature])
    
    fc_1 = Dense(units=256, activation='relu', use_bias=True, kernel_regularizer=l2(0.001))(feature)
    fc_1 = Dropout(0.5)(fc_1)
    fc_2 = Dense(units=128, activation='relu', use_bias=True)(fc_1)
    fc_3 = Dense(units=96, activation='relu', use_bias=True)(fc_2)
    fc_4 = Dense(units=4, activation='softmax', use_bias=True, name='main_output')(fc_3)
    network = Model(inputs=[up,down], outputs=fc_4)
    return network

def create_model(dim, n_sequence, n_channels, n_joint, n_output):

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

# model = create_2stream_model(20,25)
model = create_model((224,224), 8, 3, 12, 4)
print(model.summary())
