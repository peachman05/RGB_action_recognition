import cv2
import numpy as np
from model_ML import create_model_pretrain
import time
from data_helper import calculateRGBdiff

dim = (224,224)
n_sequence = 6 #8
n_channels = 3
n_output = 5
# weights_path = 'pretrain/MobileNetV2-BKB-Add3StandSideView-04-0.97-0.94.hdf5'
# weights_path = 'KARD-aug-RGBdif-40-0.92-0.98.hdf5'
#weights_path = 'pretrain/BUPT-2d-equalsplit-RGBdif-72-0.98-0.90_finalBUPT.hdf5' 
weights_path = 'BUPT-RGB-Crop-alpha-035-210-0.93-0.92.hdf5'  

### load model
model = create_model_pretrain(dim, n_sequence, n_channels, n_output, 'MobileNetV2')
model.load_weights(weights_path)

frame_window = np.empty((0, *dim, n_channels)) # seq, dim0, dim1, channel

### State Machine Define
RUN_STATE = 0
WAIT_STATE = 1
SET_NEW_ACTION_STATE = 2
state = RUN_STATE # 
previous_action = -1 # no action
text_show = 'no action'

class_text = ['run','sit','stand','walk','standup']

## fill out noise
threshold = 6
predict_queue = np.array([3] * threshold)
action_now = 1 # stand

cap = cv2.VideoCapture(0) 

start_FPS_time = time.time()
while(cap.isOpened()):
    ret, frame = cap.read()  
    
    if ret == True:
        
        new_f0 = cv2.resize(frame, dim)
        new_f0 = new_f0/255.0
        new_f = np.reshape(new_f0, (1, *new_f0.shape))
        frame_window = np.append(frame_window, new_f, axis=0)                
        if frame_window.shape[0] >= n_sequence:
            frame_window_dif = calculateRGBdiff(frame_window.copy(),0)
            frame_window_new = frame_window_dif.reshape(1, *frame_window_dif.shape)            
            # frame_window_new = frame_window.reshape(1, *frame_window.shape)
            result = model.predict(frame_window_new)
            output = result[0]
            predict_ind = np.argmax(output)
            # print("action:", predict_ind)

            # class_label = ['sit','stand','standup']
            # class_label = ['a01','a02','a03','a04','a13','a14']
            # class_label = ['dribble','shoot','pass','stand']
            
            # class_label = ['run','walk','stand']

            ## fill out noise
            # predict_queue[:-1] = predict_queue[1:]
            # predict_queue[-1] = predict_ind
            # counts = np.bincount(predict_queue)
            # if np.max(counts) >= threshold:
            #     action_now = np.argmax(counts)

            if output[predict_ind] < 0.55:
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
                if dif_time > 0.5: # wait 0.5 second
                    state = RUN_STATE
                    previous_action = new_action

            # if False:#result[0,predict_ind] < 0.4:
            #     print('no action')
            # else:
            #     # print( "{: <8}  {: <8} {:.2f}".format(class_label[predict_ind],
            #     #                 class_label[action_now],result[0,predict_ind] ) )
            #     # print("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(result[0,0],
            #     #                 result[0,1],result[0,2],result[0,3],result[0,4]))
            #     # print("{:.2f} {:.2f} {:.2f}".format(result[0,0],
            #     #                 result[0,1],result[0,2]))
            #     print(class_label[predict_ind])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text_show, (10,450), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            frame_window = frame_window[1:n_sequence]
            # vis = np.concatenate((new_f0, frame_window_new[0,n_sequence-1]), axis=0)
            # cv2.imshow('Frame', vis)
            cv2.imshow('Frame', frame)

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