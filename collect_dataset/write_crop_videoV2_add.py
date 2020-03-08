from imageai.Detection import ObjectDetection
import os
import tensorflow as tf
import cv2
import numpy as np


execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
# detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "pretrain/yolo-tiny.h5"))
# detector.setModelPath( os.path.join(execution_path , "pretrain/yolo.h5"))
detector.loadModel(detection_speed="flash") #"normal"(default), "fast", "faster" , "fastest" and "flash".
custom_objects = detector.CustomObjects(person=True)

def walk2(dirname):
    list_ = []
    for root, dirs, files in os.walk(dirname):
        # print(root)
        for filename in files:
            list_.append(os.path.join(root, filename))
    return list_

# base_path = 'F:\\Master Project\\'
base_path = '/content/gdrive/My Drive/Colab Notebooks/'    
# path_dataset = 'F:\\Master Project\\Dataset\\sit_stand\\'
path_dataset = base_path+'/Dataset/KARD-split/'
# path_save_dataset = 'F:\\Master Project\\Dataset\\sit_stand_crop02\\peach\\'
path_save_dataset = base_path+'/Dataset/KARD-split_crop/'
# action_list = ['run','sit','stand','standup','walk']
action_list = ['a01','a02','a03','a04','a05','a06','a07','a08','a09',
               'a10','a11','a12','a13','a14','a15','a16','a17','a18']
# action_list = ['run','walk']
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

        ### Read
        cap = cv2.VideoCapture(path_file)        
        length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Crop setting
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
        length_x_q = []
        length_y_q = []
        max_length = 30
        mid_x_q = []
        mid_y_q = []
        max_mid = 7

        ### Write
        if os.path.exists(path_save_file):
            print(path_save_file,' is exits!!!!')
            count += 1
            continue
        out = cv2.VideoWriter(path_save_file, fourcc, 30.0, (224,224) )
        

        for i in range(length_file):
            ret, frame = cap.read()

            detect_image, detections, extract_picture = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
                                        input_type="array", input_image=frame, output_type='array', 
                                        minimum_percentage_probability=30, extract_detected_objects = True )
            
            
            if len(extract_picture) > 0:                
                # choose picture
                max_prob = 0
                max_idx = 0
                for i,eachObject in enumerate(detections):
                    if eachObject["percentage_probability"] > max_prob:
                        max_prob = eachObject["percentage_probability"]
                        max_idx = i

                x1,y1,x2,y2 = detections[max_idx]["box_points"]
                mid_x = int((x1+x2)/2)
                mid_y = int((y1+y2)/2)
                length_x = mid_x - x1
                length_y = mid_y - y1

                ### Mean length
                length_x_q.append(length_x)
                length_y_q.append(length_y)
                if len(length_x_q) >= max_length:
                    length_x_q.pop(0)  # dequeue
                    length_y_q.pop(0)  # dequeue
                new_length_x = int( sum(length_x_q)/len(length_x_q) )
                new_length_y = int( sum(length_y_q)/len(length_y_q) )

                ### Mean mid
                mid_x_q.append(mid_x)
                mid_y_q.append(mid_y)
                if len(mid_x_q) >= max_mid:
                    mid_x_q.pop(0)  # dequeue
                    mid_y_q.pop(0)  # dequeue
                new_mid_x = sum(mid_x_q)/len(mid_x_q) 
                new_mid_y = sum(mid_y_q)/len(mid_y_q)

                new_x1 = max( int( new_mid_x - new_length_x - new_length_x*0.3 ), 0 )
                new_x2 = min( int( new_mid_x + new_length_x + new_length_x*0.3 ), width-1 )
                new_y1 = max( int( new_mid_y - new_length_y - new_length_y*0.3 ), 0 )
                new_y2 = min( int( new_mid_y + new_length_y + new_length_y*0.3 ), height-1 )               
                # print(new_x1, new_x2, new_y1, new_y2)
                ## Write image
                crop_img = frame[new_y1:new_y2, new_x1:new_x2]
                new_image = cv2.resize(crop_img, dim)
                out.write(new_image)

        out.release()
        cap.release()

        count += 1
        print('files: {:}/{:}'.format(count,n_files))

    # cv2.imshow('Frame',new_image)
    # cv2.waitKey(15)

print("finish!")