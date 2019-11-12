import numpy as np

from data_gen import DataGenerator
from data_helper import readfile_to_dict
from model_ML import create_model_pretrain

from tensorflow.python.keras.callbacks import ModelCheckpoint

dim = (224,224)
n_sequence = 15
n_channels = 3
n_output = 4
# path_dataset = 'F:\\Master Project\\Dataset\\KARD-split-frames\\'
path_dataset = 'F:\\Master Project\\Dataset\\UCF-101-Temp-frames\\'
detail_weight = 'mobileNet-UCF-15ts'

train_txt = "dataset_list/trainlistUCF.txt"
test_txt = "dataset_list/testlistUCF.txt"

# Parameters
params = {'dim': dim,
          'batch_size': 1,
        #   'n_classes': 6,
          'n_sequence': n_sequence,
          'n_channels': n_channels,
          'path_dataset': path_dataset,
          'shuffle': True}

train_d = readfile_to_dict(train_txt)
test_d = readfile_to_dict(test_txt)

# train_d = readfile_to_dict("trainlistUCF.txt")
# test_d = readfile_to_dict("testlistUCF.txt")

# Datasets
partition = {'train': list(train_d.keys()), 'validation': list(test_d.keys()) } # IDs

labels = train_d.copy()
labels.update(test_d) # Labels 

# # Generators
training_generator = DataGenerator(partition['train'] , labels, **params, type_gen='train')
validation_generator = DataGenerator(partition['validation'] , labels, **params, type_gen='test')

# # Design model
model = create_model_pretrain(dim, n_sequence, n_channels, n_output, 'MobileNet')



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
                    epochs=200,
                    initial_epoch=start_epoch,  
                    callbacks=callbacks_list)