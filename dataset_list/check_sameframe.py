import os
import cv2
import numpy as np

def walk2(dirname):
    list_ = []
    for root, dirs, files in os.walk(dirname):
        for filename in files:
            list_.append(os.path.join(root, filename))
    return list_


set_number = 2

# path_dataset = 'F:\\Master Project\\Dataset\\BUPT-dataset\\RGBdataset0'+str(set_number)+'\\'
path_dataset = 'F:\\Master Project\\Dataset\\BUPT-dataset\\RGBdataset\\'

list_file = walk2(path_dataset)
print(len(list_file))
for file_path in list_file:
    name, extension = file_path.split('.') # [0] is path/filename, [1] is extension

    if extension == 'mp4':
        cap = cv2.VideoCapture(file_path)
        length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get length of frames
        print(file_path,'vdo:',length_file)
    elif extension == 'npy':
        data = np.load(file_path)
        print(file_path,'npy:',data.shape[0])


    