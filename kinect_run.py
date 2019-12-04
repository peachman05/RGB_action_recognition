from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np
from model_ML import create_model_pretrain, create_model_skeleton, create_2stream_model
import time
from collect_dataset.skeleton_helper import read_skeleton

dim = (224,224)
# n_sequence = 15
n_sequence = 8  # KARD
n_channels = 3
n_output = 5
# weights_path = 'pretrain/MobileNetV2-BKB-Add3StandSideView-04-0.97-0.94.hdf5' 
# weights_path = 'BUPT4a-12-0.99-0.98.hdf5'
# weights_path = 'Skeleton-01-0.87-0.92.hdf5'
weights_path = 'pretrain/BUPT-2stream-22-0.98-0.97.hdf5'
n_joint = 25
type_run = '2stream'

run_2stream = (type_run == '2stream')
run_rgb = (type_run == 'rgb')
run_skeleton = (type_run == 'skelton')

### load model
if run_rgb:
    model = create_model_pretrain(dim, n_sequence, n_channels, n_output, 'MobileNetV2')
elif run_skeleton:
    model = create_model_skeleton(n_sequence, n_joint, n_output)
elif run_2stream:
    model = create_2stream_model(dim, n_sequence, n_channels, n_joint, n_output)
model.load_weights(weights_path)

frame_window = np.empty((0, *dim, n_channels)) # seq, dim0, dim1, channel
frame_sk_window = np.empty((0, n_joint*3))

# X = np.random.randint(0, 100, size=(1, n_sequence, 224, 224, 3))
## kinect
_kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

## fill out noise
threshold = 4
predict_queue = np.array([3] * threshold)
action_now = 3 # stand

start_time = time.time()

run_2stream = (type_run == '2stream')
run_rgb = (type_run == 'rgb')
run_skeleton = (type_run == 'skelton')

while(True):
    joints_data = read_skeleton(_kinect)    
    if joints_data !=  None and _kinect.has_new_color_frame():
        # get color frame
        frame = _kinect.get_last_color_frame()
        frame = frame.astype(np.uint8)
        frame = np.reshape(frame, (1080, 1920, 4))
        frame = frame[:,:,0:3]
        # append 
        frame_re = cv2.resize(frame, dim)

        if run_2stream or run_rgb:
            
            new_f = frame_re/255.0
            new_f = np.reshape(new_f, (1, *new_f.shape))
            frame_window = np.append(frame_window, new_f, axis=0)
            length_window = frame_window.shape[0]
            
        if run_2stream or run_skeleton:
            # get skeleton frame
            x, y, z = joints_data
            new_sk_f = np.array([x,y,z])
            new_sk_f = np.reshape(new_sk_f, (1, n_joint*3))
            # append 
            frame_sk_window = np.append(frame_sk_window, new_sk_f, axis=0)
            length_window = frame_sk_window.shape[0]


        if length_window >= n_sequence:
            
            if run_rgb:
                X = frame_window.reshape(1, *frame_window.shape)                
            elif run_skeleton:
                X = frame_sk_window.reshape(1, *frame_sk_window.shape)
            elif run_2stream:
                first_stream = frame_window.reshape(1, *frame_window.shape)
                second_stream = frame_sk_window.reshape(1, *frame_sk_window.shape)
                X = [first_stream,second_stream]
                

            result = model.predict(X)
            v_ = result[0]
            predict_ind = np.argmax(v_)
            # class_label = ['dribble','shoot','pass','stand']
            class_label = ['run','sit','stand','standup','walk']
            # print("action:", class_label[predict_ind])

            ## fill out noise
            predict_queue[:-1] = predict_queue[1:]
            predict_queue[-1] = predict_ind
            counts = np.bincount(predict_queue)
            if np.max(counts) >= threshold:
                action_now = np.argmax(counts)
            print( "{: <8}  {: <8}".format(class_label[predict_ind], class_label[action_now] ) )
            
            if run_rgb or run_2stream:
                frame_window = frame_window[1:n_sequence]
            
            if run_skeleton or run_2stream:
                frame_sk_window = frame_sk_window[1:n_sequence]
                    
        cv2.imshow('Frame',frame_re)

        # end_time = time.time()
        # diff_time =end_time - start_time
        # print("FPS:",1/diff_time)
        # start_time = end_time
    else:
        print('Cannot detect skeleton. Please move')

    if cv2.waitKey(15) & 0xFF == ord('q'):
        break 
    