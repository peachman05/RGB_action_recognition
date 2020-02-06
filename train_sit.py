import numpy as np

from data_gen_bkb import DataGeneratorBKB
from model_ML import create_model_pretrain
from data_helper import readfile_to_dict

from keras.callbacks.callbacks import Callback
from tensorflow.python.keras.callbacks import ModelCheckpoint

dim = (120,120)
n_sequence = 14 #6
n_channels = 3
n_output = 5    

base_path = 'F:\\Master Project\\'
# base_path = 'D:\\Peach\\'
# path_dataset = base_path + 'Dataset\\sit_stand\\'
# path_dataset = base_path + 'Dataset\\BUPT-dataset\\RGBdataset_crop\\'
path_dataset = base_path + 'Dataset\\BUPT-dataset\\RGBdataset\\'
# path_dataset = base_path + 'Dataset\\sit_stand_crop02\\'
# path_dataset = base_path + 'Dataset\\KARD-split\\'
# detail_weight = 'BUPT-2d-equalsplit-RGBdif-half' # final BUPT single person
#detail_weight = 'BUPT-RGB-Crop-alpha035-dataset02-fixaugment'#'BUPT-RGB-Crop-alpha-addDTS05'
detail_weight = 'BUPT-Conv3D-RGBdiff-dataset02'

params = {'dim': dim,
          'batch_size': 4,
          'n_sequence': n_sequence,
          'n_channels': n_channels,
          'path_dataset': path_dataset,
          'option': 'RGBdiff',
          'shuffle': True}

# train_txt = "dataset_list/trainlistSit.txt"
# test_txt = "dataset_list/testlistSit.txt"
train_txt = "dataset_list/trainlistBUPT_crop02.txt"
test_txt = "dataset_list/testlistBUPT_crop02.txt"
# train_txt = "dataset_list/trainlistKARD.txt"
# test_txt = "dataset_list/testlistKARD.txt"

train_d = readfile_to_dict(train_txt)
test_d = readfile_to_dict(test_txt)
# print(train_d) 

# Prepare key
train_keys = list(train_d.keys()) * 1  # duplicate 100 time
test_keys = list(test_d.keys()) * 4
# test_keys = list(train_d.keys()) * 500

# Label
labels = train_d.copy()
labels.update(test_d) # Labels 

# # Generators
training_generator = DataGeneratorBKB(train_keys, labels, **params, type_gen='train')
validation_generator = DataGeneratorBKB(test_keys, labels, **params, type_gen='test')

# # Design model
model = create_model_pretrain(dim, n_sequence, n_channels, n_output, "MobileNetV2")
 
load_model = False
start_epoch = 0
if load_model:
    # weights_path = 'pretrain/mobileNetV2-BKB-3ds-48-0.55.hdf5' 
    weights_path = 'BUPT-RGB-Crop-alpha035-dataset02-fixaugment-27-0.72-0.29.hdf5' #'KARD-aug-RGBdif-01-0.13-0.17.hdf5'   
    start_epoch = 27
    model.load_weights(weights_path)

## Set callback
validate_freq = 3
filepath= detail_weight+"-{epoch:02d}-{accuracy:.2f}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=False, period=validate_freq)
callbacks_list = [checkpoint]#[ PlotCallbacks()]

# # Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=300,
                    callbacks=callbacks_list,   
                    max_queue_size=1,
                    initial_epoch=start_epoch,                 
                    validation_freq=validate_freq,
                    # workers=0
                    )

