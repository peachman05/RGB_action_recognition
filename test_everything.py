from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout

import numpy as np

dim = (120,120)
X = np.zeros((1,14,dim[0],dim[1],3))
n_output = 5

model = Sequential()
model.add(Conv3D(32, kernel_size=(2, 2, 4), activation='relu',
           kernel_initializer='he_uniform', input_shape=(14,dim[0],dim[1],3))
           )
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu',
           kernel_initializer='he_uniform')
           )
           
# model.add(MaxPooling3D(pool_size=(2, 2, 2)))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(.4))
# model.add(Dense(24, activation='relu'))
# model.add(Dropout(.4))
# model.add(Dense(n_output, activation='softmax'))
print(model.summary())

print(model.predict(X).shape)
for layer in model.layers:
    weights = layer.get_weights() # list of numpy arrays
    print("-----")
    for j in range(len(weights)):
        print(weights[j].shape)

# model.add(MaxPooling3D(pool_size=(2, 2, 2)))