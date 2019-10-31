from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np

_kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)

while(True):
    if _kinect.has_new_color_frame():
        frame = _kinect.get_last_color_frame()
        frame = frame.astype(np.uint8)
        frame = np.reshape(frame, (1080, 1920, 4))
        frame = frame[:,:,0:3]