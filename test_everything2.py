from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np
import time
from collect_dataset.skeleton_helper import get_sequence_file_name, read_skeleton

path_dataset = 'F:\\Master Project\\Dataset\\BasketBall-RGB\\'
action_list = ['dribble','shoot','pass','stand']
action = action_list[2]
path_save = path_dataset +'\\'+action+'\\'+action

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
new_w = int(1920/3)
new_h = int(1080/3)
# new_w = 1920
# new_h = 1080
name_video = get_sequence_file_name(path_save,'.mp4')
out = cv2.VideoWriter(name_video, fourcc, 30.0, (new_w, new_h))

_kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

frame_all = np.empty((0, 3, 25)) # seq, dim0, dim1, channel

def save_video(_kinect):        
    frame = _kinect.get_last_color_frame()    
    frame = frame.astype(np.uint8)
    frame = np.reshape(frame, (1080, 1920, 4))
    frame = frame[:,:,0:3]   
    frame_new = cv2.resize(frame, (new_w, new_h) )
    out.write(frame_new)
    cv2.imshow('frame',frame_new)   


start_time = time.time()
frame_count = 0
while(1):    

    # Append skeleton to numpy
    joints_data = read_skeleton(_kinect)    
    if joints_data !=  None:
        
        # Write data to video
        if _kinect.has_new_color_frame():
            save_video(_kinect)

        #skeletal data
        x, y, z = joints_data
        new_f = np.array([x,y,z])
        new_f = np.reshape(new_f, (1, *new_f.shape))
        frame_all = np.append(frame_all, new_f, axis=0 )
        frame_count += 1
    else:
        print('Cannot detect skeleton. Please move')
        
        
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break 

    dif_t = (time.time() - start_time)
    if dif_t > 0:
        print("frame:{:} FPS: {:.2f}".format(frame_count, 1.0 / dif_t), end='\r')
    start_time = time.time()

    
    

# Sa
name_np_file = get_sequence_file_name(path_save,'.npy')
np.save(name_np_file, frame_all)
out.release()
print('save',name_video)
print('save',name_np_file)

cv2.destroyAllWindows()
_kinect.close()
print("finish")