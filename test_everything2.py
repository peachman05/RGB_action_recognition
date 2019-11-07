from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np
import time
from collect_dataset.skeleton_helper import get_sequence_file_name

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

def save_video(_kinect):    
    
    frame = _kinect.get_last_color_frame()
    
    frame = frame.astype(np.uint8)
    frame = np.reshape(frame, (1080, 1920, 4))
    frame = frame[:,:,0:3]      
    # frame_new = frame
    # start_time = time.time()
    frame_new = cv2.resize(frame, (new_w, new_h) )
    # dif_t = (time.time() - start_time)
    # print(dif_t)
    out.write(frame_new)
    
    
    cv2.imshow('frame',frame_new)   

start_time = time.time()
while(1):

    

    if _kinect.has_new_color_frame():
        save_video(_kinect)

    

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break 

    dif_t = (time.time() - start_time)
    # # print(dif_t)
    if dif_t > 0:
        print("FPS: {:.2f}".format(1.0 / dif_t), end='\r')
        # print('dif time', dif_t)
    start_time = time.time()
    