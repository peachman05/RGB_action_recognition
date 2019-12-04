import numpy as np

from data_gen_bkb import DataGeneratorBKB
from model_ML import create_model_skeleton
from data_helper import readfile_to_dict

from keras.callbacks.callbacks import Callback
from tensorflow.python.keras.callbacks import ModelCheckpoint

n_sequence = 8
n_output = 4
n_joint = 25
select_joint = np.array(range(25))# only arm

path_dataset = 'F:\\Master Project\\Dataset\\BUPT-dataset\\RGBdataset\\'
detail_weight = 'Skeleton'

params = {'batch_size': 32,
          'n_sequence': n_sequence,
          'path_dataset': path_dataset,
          'select_joint': select_joint,
          'shuffle': True}

train_txt = "dataset_list/trainlistBUPT.txt"
test_txt = "dataset_list/testlistBUPT.txt"
train_d = readfile_to_dict(train_txt)
test_d = readfile_to_dict(test_txt)
print(train_d)

# Prepare key
train_keys = list(train_d.keys()) * 800  # duplicate 100 time
test_keys = list(test_d.keys()) * 1000
# test_keys = list(train_d.keys()) * 500

# Label
labels = train_d.copy()
labels.update(test_d) # Labels 

# Generators
training_generator = DataGeneratorBKB(train_keys , labels, **params, type_gen='train', type_model='skeleton')
validation_generator = DataGeneratorBKB(test_keys , labels, **params, type_gen='test', type_model='skeleton')

# Design model
model = create_model_skeleton(n_sequence, n_joint, n_output)

load_model = False
start_epoch = 0
if load_model:
    weights_path = 'pretrain/Skeleton-12-0.93-0.92.hdf5' 
    # weights_path = 'Skeleton-06-0.86-0.92.hdf5'   
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

