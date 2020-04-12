import numpy as np
from model_ML import create_model_pretrain, create_model_Conv3D
import time

dim = (120,120)#(224,224)
n_sequence = 10 #6
n_channels = 3
n_output = 5

weights_path = 'BUPT-Conv3D-RGB-crop-24hidden-610-0.84-0.5525.hdf5'
# weights_path = 'BUPT-LSTM-RGB-crop-24hidden-240-0.97-0.4825.hdf5'

### load model
# model = create_model_pretrain(dim, n_sequence, n_channels, n_output, 0.35)
model = create_model_Conv3D(dim, n_sequence, n_channels, n_output)
model.load_weights(weights_path)

print(model.count_params())

X = np.random.rand(1,n_sequence,dim[0],dim[1],3)
Y = model.predict(X)

a = []
for i in range(10):
    start_time = time.time()
    Y = model.predict(X)
    a.append(time.time()-start_time)
    print('time: ', time.time()-start_time)

use_time = sum(a)/10
print(use_time)
print(1/use_time)