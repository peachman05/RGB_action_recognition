import numpy as np
import os

def get_sequence_file_name(name_file, extension):
    count = 0
    while True:
        new_name = name_file + "{:02d}".format(count)
        count += 1
        if not os.path.exists(new_name+extension):
            break    
    return new_name + extension

def read_skeleton(kinect):
    if kinect.has_new_body_frame(): 
        bodies = kinect.get_last_body_frame()        
        if bodies is not None:             
            for i in range(0, kinect.max_body_count):
                body = bodies.bodies[i]
                if not body.is_tracked: 
                    continue                 
                joints = body.joints   
                x = np.array([0.0]*25)
                y = np.array([0.0]*25)
                z = np.array([0.0]*25)

                for j in range(25): # _Joint
                    coor_point = joints[j].Position  
                    x[j] = coor_point.x
                    y[j] = coor_point.y #+ 1.1 # for adjust the tripod
                    z[j] = coor_point.z
                    
                
                return (x, y, z)   
    return None
