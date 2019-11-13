import cv2
import numpy as np
from model_ML import create_model_pretrain

dim = (224,224)
n_sequence = 8
n_channels = 3
n_output = 4
weights_path = 'pretrain/MobileNetV2-BKB-add6file-02-0.97-0.95.hdf5' 

### load model
model = create_model_pretrain(dim, n_sequence, n_channels, n_output, 'MobileNetV2')
model.load_weights(weights_path)

frame_window = np.empty((0, *dim, n_channels)) # seq, dim0, dim1, channel

cap = cv2.VideoCapture(0) 
while(cap.isOpened()):
    ret, frame = cap.read()
    
    
    
    if ret == True:
        
        new_f = cv2.resize(frame, dim)
        new_f = new_f/255.0
        new_f = np.reshape(new_f, (1, *new_f.shape))
        frame_window = np.append(frame_window, new_f, axis=0)
        if frame_window.shape[0] >= n_sequence:
            frame_window_new = frame_window.reshape(1, *frame_window.shape)
            result = model.predict(frame_window_new)
            v_ = result[0]
            predict_ind = np.argmax(v_)
            print("action:", predict_ind)

            frame_window = frame_window[1:n_sequence]

        cv2.imshow('Frame',frame)
 
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break 
    else: 
        break
 
cap.release()