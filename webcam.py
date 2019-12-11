import cv2
import numpy as np
from model_ML import create_model_pretrain
import time

dim = (224,224)
n_sequence = 8
n_channels = 3
n_output = 6
# weights_path = 'pretrain/MobileNetV2-BKB-Add3StandSideView-04-0.97-0.94.hdf5'
weights_path = 'KARD-aug-RGBdif-40-0.92-0.98.hdf5'
# weights_path = 'BUPT4a-12-0.99-0.98.hdf5' 

### load model
model = create_model_pretrain(dim, n_sequence, n_channels, n_output, 'MobileNetV2')
model.load_weights(weights_path)

frame_window = np.empty((0, *dim, n_channels)) # seq, dim0, dim1, channel

def calculateRGBdiff(sequence_img):
    'keep first frame as rgb data, other is use RGBdiff for temporal data'
    length = len(sequence_img)
    sh = sequence_img.shape
    new_sequence = np.zeros((sh[0],sh[1],sh[2],sh[3])) # (frame, w,h,3)

    # find RGBdiff frame 1 to last frame
    for i in range(length-1,0,-1): # count down
        new_sequence[i] = cv2.subtract(sequence_img[i],sequence_img[i-1])
    
    new_sequence[0] = sequence_img[0] # first frame as rgb data

    # for i in range(n_sequence):    
    #     cv2.imshow('Frame',new_sequence[i] )
    #     cv2.waitKey(500)

    return new_sequence

## fill out noise
threshold = 6
predict_queue = np.array([3] * threshold)
action_now = 1 # stand

cap = cv2.VideoCapture(0) 

start_time = time.time()
while(cap.isOpened()):
    ret, frame = cap.read()  
    
    if ret == True:
        
        new_f = cv2.resize(frame, dim)
        new_f = new_f/255.0
        new_f = np.reshape(new_f, (1, *new_f.shape))
        frame_window = np.append(frame_window, new_f, axis=0)                
        if frame_window.shape[0] >= n_sequence:
            frame_window_dif = calculateRGBdiff(frame_window)
            frame_window_new = frame_window_dif.reshape(1, *frame_window_dif.shape)            
            result = model.predict(frame_window_new)
            v_ = result[0]
            predict_ind = np.argmax(v_)
            # print("action:", predict_ind)

            # class_label = ['sit','stand','standup']
            class_label = ['a01','a02','a03','a04','a13','a14']
            # class_label = ['dribble','shoot','pass','stand']
            # class_label = ['run','sit','stand','walk']

            ## fill out noise
            predict_queue[:-1] = predict_queue[1:]
            predict_queue[-1] = predict_ind
            counts = np.bincount(predict_queue)
            if np.max(counts) >= threshold:
                action_now = np.argmax(counts)
            if result[0,predict_ind] < 0.4:
                print('no action')
            else:
                print( "{: <8}  {: <8} {:.2f}".format(class_label[predict_ind],
                                class_label[action_now],result[0,predict_ind] ) )
            # print("{:.2f} {:.2f} {:.2f} {:.2f}".format(result[0,0],
            #                 result[0,1],result[0,2],result[0,3]))
            # print("{:.2f} {:.2f} {:.2f}".format(result[0,0],
            #                 result[0,1],result[0,2]))

            frame_window = frame_window[1:n_sequence]

            cv2.imshow('Frame', frame_window_new[0,n_sequence-1])

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