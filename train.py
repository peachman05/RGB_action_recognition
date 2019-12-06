import numpy as np

from data_gen_bkb import DataGeneratorBKB
from data_helper import readfile_to_dict
from model_ML import create_model_pretrain
from keras.callbacks import ModelCheckpoint

dim = (224,224)
n_sequence = 8
n_channels = 3
n_output = 4
base_path = 'F:\\Master Project\\'
# base_path = 'D:\\Peach\\'
path_dataset = base_path + 'Dataset\\UCF-101\\'
detail_weight = 'UCF-mobilenet2'

# Parameters
params = {'dim': dim,
          'batch_size': 2,
        #   'n_classes': 6,
          'n_sequence': n_sequence,
          'n_channels': n_channels,
          'path_dataset': path_dataset,
          'shuffle': True}


train_txt = "dataset_list/UCF-101/trainlistUCF-101.txt"
test_txt = "dataset_list/UCF-101/testlistUCF-101.txt"
train_d = readfile_to_dict(train_txt)
test_d = readfile_to_dict(test_txt)

# Prepare key
train_keys = list(train_d.keys()) * 3  # duplicate 100 time
test_keys = list(test_d.keys()) * 1

# Label
labels = train_d.copy()
labels.update(test_d) # Labels 

# # Generators
training_generator = DataGeneratorBKB(train_keys, labels, **params, type_gen='train')
validation_generator = DataGeneratorBKB(test_keys, labels, **params, type_gen='test')

# # Design model
model = create_model_pretrain(dim, n_sequence, n_channels, n_output, 'MobileNetV2')



load_model = False
start_epoch = 0
if load_model:
    weights_path = 'mobileNetV2-KARD-18a-06-0.12.hdf5'    
    start_epoch = 7
    model.load_weights(weights_path)

# print(model.summary())

filepath= detail_weight+"-{epoch:02d}-{accuracy:.2f}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True)
callbacks_list = [checkpoint]
# # Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=300,
                    initial_epoch=start_epoch,  
                    callbacks=callbacks_list)