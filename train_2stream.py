import numpy as np

from data_gen_bkb import DataGeneratorBKB
from model_ML import create_model_pretrain
from data_helper import readfile_to_dict

from keras.callbacks.callbacks import Callback
from tensorflow.python.keras.callbacks import ModelCheckpoint

dim = (224,224)
n_sequence = 8
n_channels = 3
n_output = 4
n_joint = 12
select_joint = [..]
path_dataset = 'F:\\Master Project\\Dataset\\BasketBall-RGB\\'
detail_weight = 'MobileNetV2-BKB-Add3StandSideView'

params = {'dim': dim,
          'batch_size': 2,
          'n_sequence': n_sequence,
          'n_channels': n_channels,
          'path_dataset': path_dataset,
          'select_joint': select_joint,
          'shuffle': True}

train_txt = "dataset_list/trainlistBKB.txt"
test_txt = "dataset_list/testlistBKB.txt"
train_d = readfile_to_dict(train_txt)
test_d = readfile_to_dict(test_txt)
print(train_d)

# Prepare key
train_keys = list(train_d.keys()) * 40  # duplicate 100 time
test_keys = list(test_d.keys()) * 60
# test_keys = list(train_d.keys()) * 500

# Label
labels = train_d.copy()
labels.update(test_d) # Labels 

# Generators
training_generator = DataGenerator2Stream(train_keys , labels, **params, type_gen='train')
validation_generator = DataGenerator2Stream(test_keys , labels, **params, type_gen='test')

# Design model
model = create_2stream_model(dim, n_sequence, n_channels, n_joint, n_output)

load_model = True
start_epoch = 0
if load_model:
    # weights_path = 'pretrain/mobileNetV2-BKB-3ds-48-0.55.hdf5' 
    weights_path = 'pretrain/MobileNetV2-BKB-add6file-02-0.97-0.95.hdf5'   
    start_epoch = 0
    model.load_weights(weights_path)

## Set callback
validate_freq = 1
filepath= detail_weight+"-{epoch:02d}-{accuracy:.2f}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, period=validate_freq)
callbacks_list = [checkpoint]

# # Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=300,
                    callbacks=callbacks_list,
                    initial_epoch=start_epoch,                 
                    validation_freq=validate_freq
                    )

