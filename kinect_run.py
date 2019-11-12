from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np
from model_ML import create_model_pretrain_old
import time

dim = (224,224)
# n_sequence = 15
n_sequence = 8  # KARD
n_channels = 3
n_output = 4
weights_path = 'pretrain/MobileNetV2-BKB-80-0.95-0.91.hdf5' 
# weights_path = 'mobileNetV2-BKB-3ds-48-0.55.hdf5'

### load model
# model = create_model_pretrain(dim, n_sequence, n_channels, n_output)
model = create_model_pretrain(dim, n_sequence, n_channels, n_output, 'MobileNetV2')
model.load_weights(weights_path)

frame_window = np.empty((0, *dim, n_channels)) # seq, dim0, dim1, channel
X = np.random.randint(0, 100, size=(1, n_sequence, 224, 224, 3))
## kinect
_kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
start_time = time.time()
while(True):

    if _kinect.has_new_color_frame():
        frame = _kinect.get_last_color_frame()
        frame = frame.astype(np.uint8)
        frame = np.reshape(frame, (1080, 1920, 4))
        frame = frame[:,:,0:3]
        # frame_show = cv2.resize(frame, None, fx=0.5, fy=0.5)
        frame_re = cv2.resize(frame, dim)
        new_f = frame_re/255.0
        new_f = np.reshape(new_f, (1, *new_f.shape))
        frame_window = np.append(frame_window, new_f, axis=0)
        if frame_window.shape[0] >= n_sequence:
            frame_window_new = frame_window.reshape(1, *frame_window.shape)
            result = model.predict(frame_window_new)
            v_ = result[0]
            predict_ind = np.argmax(v_)
            print("action:", predict_ind)

            frame_window = frame_window[1:n_sequence]
        cv2.imshow('Frame',frame_re)
        end_time = time.time()
        diff_time =end_time - start_time
        print("FPS:",1/diff_time)
        start_time = end_time

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break 
    