import cv2
import numpy as np
from model_ML import create_model_pretrain, create_model_Conv3D
import time
from data_helper import calculateRGBdiff, calculate_intersection
from imageai.Detection import ObjectDetection
from operator import itemgetter

dim = (120,120)
n_sequence = 10
n_channels = 3
n_output = 5

# weights_path = 'BUPT-2d-equalsplit-RGBdif-72-0.98-0.90.hdf5' 
weights_path =  'BUPT-RGBdiff-crop-Conv3D-verytiny-dataset02-1160-0.85-0.79.hdf5' 
# weights_path = 'BUPT-RGBdiff-crop-Conv3D-verytiny-dataset02-1600-0.88-0.77.hdf5' 
# weights_path = 'KARD-Conv3D-RGBdiff-crop-1460-0.80-0.85.hdf5' #'KARD-LSTM-RGBdiff-crop-24hidden-220-0.87-0.9028.hdf5'
# weights_path = 'KARD-LSTM-RGBdiff-crop-24hidden-220-0.87-0.9028.hdf5'

### load model
# model = create_model_pretrain(dim, n_sequence, n_channels, n_output, 0.35)
model = create_model_Conv3D(dim, n_sequence, n_channels, n_output) 
model.load_weights(weights_path)

print(model.summary())

max_person = 10
frame_window = [np.empty((0, *dim, n_channels))]*max_person # max_person, seq, dim0, dim1, channel
cordinate = np.ones((max_person, 4)) # x1,y1,x2,y2

### State Machine Define
RUN_STATE = 0
WAIT_STATE = 1
SET_NEW_ACTION_STATE = 2
state = [RUN_STATE] * max_person # 
previous_action = [-1] * max_person  # no action
previous_prob = [0] * max_person
start_time = [0] * max_person
text_show = 'no action'

class_text = ['run','sit','stand','walk','standup']
# class_text = ['a01','a02','Two Hand Wave','a04','a05','a06','a07','a08','a09',
#                'a10','a11','a12','hand clap','a14','a15','a16','a17','a18']

### ImageAI
model_path = "pretrain/yolo-tiny.h5"
# model_path = "pretrain/yolo.h5"
detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
# detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.loadModel(detection_speed="flash") #"normal"(default), "fast", "faster" , "fastest" and "flash".
custom_objects = detector.CustomObjects(person=True)


### Main
# cap = cv2.VideoCapture("C:/Users/peachman/Desktop/walk.mp4")
cap = cv2.VideoCapture(0)

# Crop setting
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
length_x_q = [[] for i in range(max_person)]
length_y_q = [[] for i in range(max_person)]
max_length = 5
mid_x_q = [[] for i in range(max_person)]
mid_y_q = [[] for i in range(max_person)]
max_mid = 3
font = cv2.FONT_HERSHEY_SIMPLEX

start_FPS_time = time.time()
while(cap.isOpened()):
    ret, frame = cap.read()  

    if ret == True:        
        ### Detect person
        returned_image, detections, extract_image = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
                        input_type='array', input_image=frame, output_type='array',
                        minimum_percentage_probability=20,
                        extract_detected_objects = True )
        detections = sorted(detections, key=itemgetter('box_points') )
        n_person = len(detections)
        if n_person > 0:            
            print("------")
            for i in range(n_person):
                print(detections[i]["box_points"])
                x1,y1,x2,y2 = detections[i]["box_points"]

                mid_x = int((x1+x2)/2)
                mid_y = int((y1+y2)/2)
                length_x = mid_x - x1
                length_y = mid_y - y1

                ## Mean length
                length_x_q[i].append(length_x)
                length_y_q[i].append(length_y)
                if len(length_x_q[i]) >= max_length:
                    length_x_q[i].pop(0)  # dequeue
                    length_y_q[i].pop(0)  # dequeue
                new_length_x = int( sum(length_x_q[i])/len(length_x_q[i]) )
                new_length_y = int( sum(length_y_q[i])/len(length_y_q[i]) )

                ## Mean mid
                mid_x_q[i].append(mid_x)
                mid_y_q[i].append(mid_y)
                if len(mid_x_q[i]) >= max_mid:
                    mid_x_q[i].pop(0)  # dequeue
                    mid_y_q[i].pop(0)  # dequeue
                new_mid_x = sum(mid_x_q[i])/len(mid_x_q[i]) 
                new_mid_y = sum(mid_y_q[i])/len(mid_y_q[i])

                new_x1 = max( int( new_mid_x - new_length_x - new_length_x*0.5 ), 0 )
                new_x2 = min( int( new_mid_x + new_length_x + new_length_x*0.25 ), width-1 )
                new_y1 = max( int( new_mid_y - new_length_y - new_length_y*0.4 ), 0 )
                new_y2 = min( int( new_mid_y + new_length_y + new_length_y*0.2 ), height-1 )

                crop_image = frame[new_y1:new_y2, new_x1:new_x2]

                # crop_image = frame[y1:y2, x1:x2] #extract_image[i]
                new_f0 = cv2.resize(crop_image, dim)
                new_f0 = new_f0/255.0
                new_f = np.reshape(new_f0, (1, *new_f0.shape))
                frame_window[i] = np.append(frame_window[i], new_f, axis=0) 
                # print(frame_window[i].shape)               
                if frame_window[i].shape[0] >= n_sequence:
                    frame_window_dif = calculateRGBdiff(frame_window[i].copy(),0)
                    frame_window_new = frame_window_dif.reshape(1, *frame_window_dif.shape)            
                    # frame_window_new = frame_window[i].reshape(1, *frame_window[i].shape)
                    result = model.predict(frame_window_new)
                    output = result[0]
                    predict_ind = np.argmax(output)

                    ## Noise remove
                    if output[predict_ind] < 0.40:
                        new_action = -1 # no action(noise)
                    else:
                        new_action = predict_ind # action detect            

                    ### Use State Machine to delete noise between action(just for stability)
                    ### RUN_STATE: normal state, change to wait state when action is changed
                    if state[i] == RUN_STATE:
                        if new_action != previous_action[i]: # action change
                            state[i] = WAIT_STATE
                            start_time[i] = time.time() 

                    # ### WAIT_STATE: wait 0.5 second when action from prediction is change to fillout noise
                    elif state[i] == WAIT_STATE:
                        dif_time = time.time() - start_time[i]
                        if dif_time > 0.5: # wait 0.5 second
                            state[i] = RUN_STATE
                            previous_action[i] = new_action
                            previous_prob[i] = output[new_action]
                                            
                    if previous_action[i] != -1:
                        text_show = "{:}: {: <2}  {:.2f} ".format(i, class_text[previous_action[i]],
                                            previous_prob[i] )
                        # text_show = "{:}: {: <2}  {:.2f} ".format(i, class_text[new_action],
                        #                     output[new_action] )
                    else:
                        text_show = "{:}: no action ".format(i) 
                    
                    print('state:', state)


                    frame_window[i] = frame_window[i][1:n_sequence]

                    print(text_show) 
                    # text_show = "{:}: {: <5}  {:.2f} ".format(i,class_text[predict_ind],
                    #                         output[predict_ind] )
                
                cv2.rectangle(frame,(new_x1,new_y1),(new_x2,new_y2),(0,255,0),2)                
                cv2.rectangle(frame,(new_x1-1,new_y1),(new_x2+1,new_y1+40),(0,255,0),-1) # draw background text
                # cv2.putText(frame, text_show, (new_x1,new_y1+30), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, text_show, (new_x1,new_y1+30), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
                # cv2.rectangle(frame,(0,0),(300,400),(0,255,0),2)
                # intersect_area = calculate_intersection((x1,y1,x2,y2), (0,0,300,400))
                # print('intersect_area:',intersect_area)
                # cv2.putText(frame, str(intersect_area), (10,450), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)        
        else:
            print("cannot detect")

        cv2.imshow('Frame', frame) #returned_image)

        end_FPS_time = time.time()
        diff_time = end_FPS_time - start_FPS_time
        print("FPS:",1/diff_time)
        start_FPS_time = end_FPS_time

        # Press Q on keyboard to  exit
        key = cv2.waitKey(10)
        if key == ord('q'):
                break 
    else: 
        break
 
cap.release()