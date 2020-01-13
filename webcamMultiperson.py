import cv2
import numpy as np
from model_ML import create_model_pretrain
import time
from data_helper import calculateRGBdiff
from imageai.Detection import ObjectDetection

dim = (224,224)
n_sequence = 8
n_channels = 3
n_output = 5

weights_path = 'BUPT-2d-equalsplit-RGBdif-72-0.98-0.90.hdf5' 

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

### ImageAI
model_path = "pretrain/yolo-tiny.h5"
detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel(detection_speed="flash") #"normal"(default), "fast", "faster" , "fastest" and "flash".
custom_objects = detector.CustomObjects(person=True)


### Main
cap = cv2.VideoCapture(0)
start_time = time.time()
while(cap.isOpened()):
    ret, frame = cap.read()  

    if ret == True:        
        ### Detect person
        returned_image, detection, extract_image = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
                        input_type='array', input_image=frame, output_type='array',
                        minimum_percentage_probability=50,
                        extract_detected_objects = True )
        if len(extract_image) > 0:
            crop_image = extract_image[0]
            new_f0 = cv2.resize(crop_image, dim)
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

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(returned_image, text_show, (10,450), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                frame_window = frame_window[1:n_sequence]
                # vis = np.concatenate((new_f0, frame_window_new[0,n_sequence-1]), axis=0)
                # cv2.imshow('Frame', vis)
                cv2.imshow('Frame', returned_image)

        # end_time = time.time()
        # diff_time =end_time - start_time
        # print("FPS:",1/diff_time)
        # start_time = end_time

        # Press Q on keyboard to  exit
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break 
    else: 
        break
 
cap.release()