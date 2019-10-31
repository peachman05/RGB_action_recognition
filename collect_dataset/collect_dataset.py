from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.animation as animation

import cv2
import numpy as np
import time
from skeleton_helper import read_skeleton, get_sequence_file_name

##################################
############# define #############
##################################
path_dataset = 'F:\\Master Project\\Dataset\\BasketBall-RGB\\'
action_list = ['dribble','shoot','pass','stand']
action = action_list[0]
path_save = path_dataset +'\\'+action+'\\'+action

_kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
new_w = int(1920/3)
new_h = int(1080/3)

name_video = get_sequence_file_name(path_save,'.mp4')
out = cv2.VideoWriter(name_video, fourcc, 40.0, (new_w, new_h))
# f = open(action+'.txt', 'w') 
frame_all = np.empty((0, 3, 25)) # seq, dim0, dim1, channel

##################################
############# fucntion ###########
##################################
# full bonelist 25 joints
bone_list = [[24,12], [25,12], [12,11], [11,10], [10,9], # right arm
            [22,8] ,[23,8], [8,7], [7,6], [6,5], # left arm
            [4,3], [3,21], [9,21], [5,21], [21,2], [2,1], [17,1], [13,1], # body
            [17,18], [18,19], [19,20], # right leg
            [13,14], [14,15], [15,16]]

bone_list = np.array(bone_list) - 1

def save_video(_kinect):    
    frame = _kinect.get_last_color_frame()
    frame = frame.astype(np.uint8)
    frame = np.reshape(frame, (1080, 1920, 4))
    frame = frame[:,:,0:3]
    frame_new = cv2.resize(frame, (new_w, new_h) )
    out.write(frame_new)
    cv2.imshow('frame',frame_new)            

def update_lines(num, _kinect, lines, bone_list, my_ax):    
    # global start_time
    global frame_all
    # dif_t = (time.time() - start_time)
    # if dif_t > 0:
    #     print("FPS: ", 1.0 / dif_t )
    if _kinect.has_new_color_frame():
        save_video(_kinect)
    
    joints_data = read_skeleton(_kinect)    
    if joints_data !=  None:
        #skeletal data
        x, y, z = joints_data   

        new_f = np.array([x,y,z])
        new_f = np.reshape(new_f, (1, *new_f.shape))
        frame_all = np.append(frame_all, new_f, axis=0 )
        # f.write(x, y, z]  + "\n")

        for line, bone in zip(lines, bone_list):
            # NOTE: there is no .set_data() for 3 dim data...
            line.set_data([x[bone[0]], x[bone[1]]], [z[bone[0]], z[bone[1]]])
            line.set_3d_properties([y[bone[0]], y[bone[1]]])

        for i, t in enumerate(annots):
            x_, y_, _ = proj3d.proj_transform(x[i], z[i], y[i], my_ax.get_proj())
            t.set_position((x_,y_))
            t.set_text(str(i+1))

    
    # start_time = time.time()

    return lines, annots

##################################
########### plot graph ###########
##################################
num_joint = 25
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.legend()
ax.set_xlim3d(-0.8, 0.8)
ax.set_ylim3d(0, 4) # z-kinect
ax.set_zlim3d(-1.4, 0.5) # y-kinect
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('y')
annots = [ax.text2D(0,0,"POINT") for _ in range(num_joint)]

x = np.array(range(num_joint))
y = np.array(range(num_joint))
z = np.array(range(num_joint))
lines = [ax.plot([x[bone[0]], x[bone[1]]],
                 [z[bone[0]], z[bone[1]]],
                 [y[bone[0]], y[bone[1]]])[0] for bone in bone_list]


##################################
############## main ##############
##################################
#countdown
for i in range(5,0,-1):
    print("start in:", i)
    time.sleep(1)
    

line_ani = animation.FuncAnimation(fig, update_lines, None,
                                   fargs=(_kinect, lines, bone_list, ax),
                                   interval=1, blit=False)


# loop
plt.show()

# close
name_np_file = get_sequence_file_name(path_save,'.npy')
np.save(name_np_file, frame_all)

out.release()
cv2.destroyAllWindows()
_kinect.close()
print("finish")