from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np
import time
from skeleton_helper import get_sequence_file_name, read_skeleton

# Parameter
path_dataset = 'F:\\Master Project\\Dataset\\BasketBall-RGB\\'
action_select = 3 # 0=dribble, 1=shoot, 2=pass, 3=stand
auto_exit = False # (optional) Auto save and close after ... second
run_time = 10 # second


action_list = ['dribble','shoot','pass','stand']
action = action_list[action_select]
path_save = path_dataset +'\\'+action+'\\'+action
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
new_w = int(1920/3)
new_h = int(1080/3)
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
original_time = start_time + 1000
frame_count = 0
while(1):    

    # Check camera can detect body or not
    joints_data = read_skeleton(_kinect)    
    if joints_data !=  None:

        if frame_count == 0:
            original_time = start_time
        
        # Write data to video
        if _kinect.has_new_color_frame():
            save_video(_kinect)

        # Append skeleton data to numpy array
        x, y, z = joints_data
        new_f = np.array([x,y,z])
        new_f = np.reshape(new_f, (1, *new_f.shape))
        frame_all = np.append(frame_all, new_f, axis=0 )

        frame_count += 1
    else:
        print('Cannot detect skeleton. Please move')
        
    # calculate FPS
    dif_t = (time.time() - start_time)
    pass_t = time.time() - original_time
    if dif_t > 0 and frame_count > 0:
        print("frame:{:} FPS: {:.2f} time: {:02}m {:02}s".format(frame_count, 1.0 / dif_t, int(pass_t/60), int(pass_t)%60), end='\r')
        # print("frame:{:} FPS: {:.2f}".format(frame_count, 1.0 / dif_t), end='\r')
    start_time = time.time()
            
        
    if pass_t > run_time and auto_exit:
        print('time out!!!!') 
        break 
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break 

    
# Save file
name_np_file = get_sequence_file_name(path_save,'.npy')
np.save(name_np_file, frame_all)
out.release()
print('save',name_video)
print('save',name_np_file)

cv2.destroyAllWindows()
_kinect.close()
print("finish")