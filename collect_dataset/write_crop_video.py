from imageai.Detection import ObjectDetection
import os
import tensorflow as tf
import cv2
import numpy as np


execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
# detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath( os.path.join(execution_path , "../pretrain/yolo.h5"))
detector.loadModel(detection_speed="fast")

def walk2(dirname):
    list_ = []
    for root, dirs, files in os.walk(dirname):
        # print(root)
        for filename in files:
            list_.append(os.path.join(root, filename))
    return list_

    
path_dataset = 'F:\\Master Project\\Dataset\\sit_stand\\'
path_save_dataset = 'F:\\Master Project\\Dataset\\sit_stand_crop\\'
action_list = ['sit','stand','standup']
dim = (224,224)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')

# count file
n_files = len(walk2(path_dataset))
count = 0
for i in range(len(action_list)):
    path_folder = path_dataset + action_list[i]
    path_save_folder = path_save_dataset + action_list[i]
    list_file = walk2(path_folder)
    for file_path in list_file:        
        name_file = file_path.split('\\')[-1]

        path_file = path_folder+'\\'+name_file
        path_save_file = path_save_folder+'\\'+name_file 

        # read
        cap = cv2.VideoCapture(path_file)
        length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # write
        if os.path.exists(path_save_file):
            print(path_save_file,' is exits!!!!')
            continue
        out = cv2.VideoWriter(path_save_file, fourcc, 30.0, (224,224) )

        for i in range(length_file):
            ret, frame = cap.read()

            detect_image, detections, extract_picture = detector.detectObjectsFromImage(input_type="array", input_image=frame, output_type='array', 
                                                 minimum_percentage_probability=10, extract_detected_objects = True )
            
            # choose picture
            max_prob = 0
            max_idx = 0
            for i,eachObject in enumerate(detections):
                if eachObject["name"] == 'person' and eachObject["percentage_probability"] > max_prob:
                    max_prob = eachObject["percentage_probability"]
                    max_idx = i
                    # print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ",
                    # eachObject["box_points"] )
                    # print("--------------------------------")
            if max_idx >= len(extract_picture):
                # if no detection, use black array
                crop_img = np.zeros((*dim, 3))
            else:
                crop_img = extract_picture[max_idx]
            new_image = cv2.resize(crop_img, dim)

            out.write(new_image)

        out.release()
        cap.release()

        count += 1
        print('files: {:}/{:}'.format(count,n_files))

    # cv2.imshow('Frame',new_image)
    # cv2.waitKey(15)

print("finish!")