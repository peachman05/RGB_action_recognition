import cv2
import numpy as np
from model_ML import create_model_pretrain
import time
from data_helper import calculateRGBdiff
from imageai.Detection import ObjectDetection

dim = (224,224)
n_sequence = 6
n_channels = 3
n_output = 5

# weights_path = 'BUPT-2d-equalsplit-RGBdif-72-0.98-0.90.hdf5' 
#weights_path =  'BUPT-RGB-Crop-96-0.92-0.88.hdf5' 
weights_path = 'BUPT-RGB-Crop-alpha-035-210-0.93-0.92.hdf5' 
### load model
model = create_model_pretrain(dim, n_sequence, n_channels, n_output, 'MobileNetV2')
model.load_weights(weights_path)

print(model.summary())

frame_window = np.empty((0, *dim, n_channels)) # seq, dim0, dim1, channel

### State Machine Define
RUN_STATE = 0
WAIT_STATE = 1
SET_NEW_ACTION_STATE = 2
state = RUN_STATE # 
previous_action = -1 # no action
text_show = 'no action'

class_text = ['run','sit','stand','walk','standup']

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
cap = cv2.VideoCapture("C:/Users/peachman/Desktop/walk06_03.mp4")

# Crop setting
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
length_x_q = []
length_y_q = []
max_length = 30
mid_x_q = []
mid_y_q = []
max_mid = 6

cap.set(cv2.CAP_PROP_POS_FRAMES, 500)

start_FPS_time = time.time()
while(cap.isOpened()):
    ret, frame = cap.read()  

    if ret == True:        
        ### Detect person
        returned_image, detections, extract_image = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
                        input_type='array', input_image=frame, output_type='array',
                        minimum_percentage_probability=20,
                        extract_detected_objects = True )
        if len(extract_image) > 0:

            # choose picture
            max_prob = 0
            max_idx = 0
            for i,eachObject in enumerate(detections):
                if eachObject["percentage_probability"] > max_prob:
                    max_prob = eachObject["percentage_probability"]
                    max_idx = i

            #### Mean 
            x1,y1,x2,y2 = detections[max_idx]["box_points"]
            mid_x = int((x1+x2)/2)
            mid_y = int((y1+y2)/2)
            length_x = mid_x - x1
            length_y = mid_y - y1

            ## Mean length
            length_x_q.append(length_x)
            length_y_q.append(length_y)
            if len(length_x_q) >= max_length:
                length_x_q.pop(0)  # dequeue
                length_y_q.pop(0)  # dequeue
            new_length_x = int( sum(length_x_q)/len(length_x_q) )
            new_length_y = int( sum(length_y_q)/len(length_y_q) )

            ## Mean mid
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

            crop_image = frame[new_y1:new_y2, new_x1:new_x2] #extract_image[max_idx]
            new_f0 = cv2.resize(crop_image, dim)
            new_f0 = new_f0/255.0
            new_f = np.reshape(new_f0, (1, *new_f0.shape))
            frame_window = np.append(frame_window, new_f, axis=0)                
            if frame_window.shape[0] >= n_sequence:
                # frame_window_dif = calculateRGBdiff(frame_window.copy(),0)
                # frame_window_new = frame_window_dif.reshape(1, *frame_window_dif.shape)            
                frame_window_new = frame_window.reshape(1, *frame_window.shape)
                result = model.predict(frame_window_new)
                output = result[0]
                predict_ind = np.argmax(output)
            

                if output[predict_ind] < 0.25:
                    new_action = -1 # no action(noise)
                else:
                    new_action = predict_ind # action detect            

                ### Use State Machine to delete noise between action(just for stability)
                ### RUN_STATE: normal state, change to wait state when action is changed
                if state == RUN_STATE:
                    if new_action != previous_action: # action change
                        state = WAIT_STATE
                        start_time = time.time()     
                    else:
                        if previous_action == -1:
                            text_show = 'no action'                                              
                        else:
                            text_show = "{: <22}  {:.2f} ".format(class_text[previous_action],
                                        output[previous_action] )
                        print(text_show)  

                ### WAIT_STATE: wait 0.5 second when action from prediction is change to fillout noise
                elif state == WAIT_STATE:
                    dif_time = time.time() - start_time
                    if dif_time > 0.05: # wait 0.5 second
                        state = RUN_STATE
                        previous_action = new_action

                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.rectangle(frame,(new_x1,new_y1),(new_x2,new_y2),(0,255,0),2)
                cv2.putText(frame, text_show, (10,450), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                frame_window = frame_window[1:n_sequence]
                # vis = np.concatenate((new_f0, frame_window_new[0,n_sequence-1]), axis=0)
                # cv2.imshow('Frame', vis)
                
        else:
            print("cannot detect")

        cv2.imshow('Frame', frame) #returned_image)

        end_FPS_time = time.time()
        diff_time = end_FPS_time - start_FPS_time
        print("FPS:",1/diff_time)
        start_FPS_time = end_FPS_time

        # Press Q on keyboard to  exit
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break 
    else: 
        break
 
cap.release()