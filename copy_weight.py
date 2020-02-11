from model_ML import create_model_Conv3D
from keras import Sequential
from keras.applications import MobileNetV2

dim = (120,120)
n_sequence = 10
n_channels = 3
n_output = 5 
detail_weight = 'BUPT-Conv3D-dataset02-transfer'   

# new_weight = np.zeros((3,3,3,3,32))

model = create_model_Conv3D(dim, n_sequence, n_channels, n_output) 

mobile_model = Sequential()
mobile_model.add(MobileNetV2(weights='imagenet',include_top=False))

weights = model.layers[0].get_weights() # first layer
weight_mobile = mobile_model.layers[0].get_weights()[0] # first layer, first weight

for i in range(3):
    weights[0][:,:,i,:,:] = weight_mobile

model.layers[0].set_weights(weights)
# model.layers[2].set_weights(weights)

model.save_weights(detail_weight+'-0-0-0.hdf5')
print('test')

