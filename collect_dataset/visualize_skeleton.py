# ---- plot graph ------
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.animation as animation
import time

import pickle

###################################################################
###---------------------  Read Data  ---------------------------###
###################################################################

# Parameter
# path_dataset = 'F:\\Master Project\\Dataset\\BasketBall-RGB\\'
path_dataset = 'F:\\Master Project\\Dataset\\BUPT-dataset\\RGBdataset02\\'
action_select = 3 # 0=dribble, 1=shoot, 2=pass, 3=stand
number_file = '00'

# action_list = ['dribble','shoot','pass','stand']
action_list = ['run', 'shoot', 'sit', 'stand', 'walk']
action = action_list[action_select]
path_save = path_dataset +'\\'+action+'\\'+action
path_file = path_save + number_file + '.npy'
print(path_file)
data = np.load(path_file)
data_plot = data 

print('shape:', data_plot.shape)



###################################################################
###---------------------  Plot Graph ---------------------------###
###################################################################

num_joint = 25
bone_list = [[24,12], [25,12], [12,11], [11,10], [10,9], # right arm
            [22,8] ,[23,8], [8,7], [7,6], [6,5], # left arm
            [4,3], [3,21], [9,21], [5,21], [21,2], [2,1], [17,1], [13,1], # body
            [17,18], [18,19], [19,20], # right leg
            [13,14], [14,15], [15,16]]
bone_list = np.array(bone_list) - 1


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
start_time = time.time()
def update_lines(num, data, lines, bone_list, my_ax): 
    global start_time
    num += 0
    dif_t = (time.time() - start_time)
    x = data[num,0]
    y = data[num,1]
    z = data[num,2]

    for line, bone in zip(lines, bone_list):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data([x[bone[0]], x[bone[1]]], [z[bone[0]], z[bone[1]]])  # x horizontal, z deep(y in matplot)
        line.set_3d_properties([y[bone[0]], y[bone[1]]]) # y vertical (z in matplot)

    for i, t in enumerate(annots):
        x_, y_, _ = proj3d.proj_transform(x[i], z[i], y[i], my_ax.get_proj())
        t.set_position((x_,y_))
        t.set_text(str(i+1))
   
    start_time = time.time()

    return lines, annots

x = np.array(range(num_joint))
y = np.array(range(num_joint))
z = np.array(range(num_joint))
lines = [ax.plot([x[bone[0]], x[bone[1]]],
                 [z[bone[0]], z[bone[1]]],
                 [y[bone[0]], y[bone[1]]])[0] for bone in bone_list]

line_ani = animation.FuncAnimation(fig, update_lines, data_plot.shape[0],
                fargs=(data_plot, lines, bone_list, ax),
                interval=1, blit=False)

# loop
plt.show()
