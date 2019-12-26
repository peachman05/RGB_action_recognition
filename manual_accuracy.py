from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import cv2

from model_ML import create_model_pretrain
from data_helper import readfile_to_dict        

dim = (224,224)
n_sequence = 8
n_channels = 3
n_output = 5
# base_path = 'D:\\Peach\\'
base_path = 'F:\\Master Project\\'
# path_dataset = base_path+'Dataset\\sit_stand\\'
path_dataset = base_path + 'Dataset\\BUPT-dataset\\RGBdataset\\'


## dataset
test_txt = "dataset_list/testlistBUPT.txt"
test_d = readfile_to_dict(test_txt)
labels = test_d.copy()
num_mul = 1
print(len(test_d.keys()))
key_list = list(test_d.keys()) * num_mul

# model
weights_path = 'BUPT-augmentKeras-RGB-72-0.88-0.85.hdf5' 
model = create_model_pretrain(dim, n_sequence, n_channels, n_output, 'MobileNetV2')
model.load_weights(weights_path)

y_pred_all = []

def get_sampling_frame(len_frames):   
    '''
    Sampling n_sequence frame from video file
    Input: 
        len_frames -- number of frames that this video have
    Output: 
        index_sampling -- n_sequence frame indexs from sampling algorithm 
    '''

    # Define maximum sampling rate
    sample_interval = len_frames//n_sequence
    start_i = 0 #np.random.randint(0, len_frames - sample_interval * self.n_sequence + 1)

    # if self.type_gen =='train':
    #     random_sample_range = 7
    #     if random_sample_range*self.n_sequence > len_frames:
    #         random_sample_range = len_frames//self.n_sequence

    #     if random_sample_range <= 0:
    #         print('test:',random_sample_range, len_frames)
    #     # Randomly choose sample interval and start frame
    #     sample_interval = np.random.randint(3, random_sample_range + 1)
        
    #     start_i = np.random.randint(0, len_frames - sample_interval * self.n_sequence + 1)
    
    # Get n_sequence index of frames
    index_sampling = []
    end_i = sample_interval * n_sequence + start_i
    for i in range(start_i, end_i, sample_interval):
        if len(index_sampling) < n_sequence:
            index_sampling.append(i)
    
    return index_sampling


X1 = np.empty((1, n_sequence, *dim, n_channels))
for i, ID in enumerate(key_list):  # ID is name of file (2 batch)
    path_video = path_dataset + ID + '.mp4'
    print(path_video)

    cap = cv2.VideoCapture(path_video)    
    length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get how many frames this video have ,-1 because some bug          
    index_sampling = get_sampling_frame(length_file) # get sampling index  
    print(index_sampling)
                    
    # Get RGB sequence
    for j, n_pic in enumerate(index_sampling):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n_pic) # jump to that index
        ret, frame = cap.read()
        new_image = cv2.resize(frame, dim)          
        X1[0,j,:,:,:] = new_image
    
    cap.release()
    X1[0,] = X1[0,]/255.0
    # if option == 'RGBdiff':
    #     X1[i,] = calculateRGBdiff(X1[i,])                        
    y_pred_prob = model.predict(X1)
    y_pred = np.argmax(y_pred_prob[0])
    y_pred_all.append(y_pred)


test_y = np.array(list(test_d.values()) * num_mul)
normalize = True

all_y = len(test_y)
for i in range(len(y_pred_all)):
    if test_y[i] != y_pred_all[i]:
        print(key_list[i],' actual:',test_y[i],'predict:',y_pred_all[i])

cm = confusion_matrix(test_y, y_pred_all)
if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
else:
    print('Confusion matrix, without normalization')

print(cm)
# if len(cm) < 4:
sum = 0.0
for i in range(n_output):
    sum += cm[i,i]

# (cm[0,0] + cm[1,1] + cm[2,2] + cm[3,3] )

accuracy = sum / n_output
# # accuracy = (cm[0,0] + cm[1,1]) / 4
print("accuracy:",accuracy)

# classes = ['ApplyEyeMakeup','Archery','BabyCrawling','Basketball']
# classes = ['0','1','2','3']
# classes = ['sit','stand','standup']
# classes = ['run','sit','stand','walk']
classes = ['run','sit','stand','walk', 'standup']

df_cm = pd.DataFrame(cm, columns=classes, index=classes)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
fig, ax = plt.subplots(figsize=(5,5))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,fmt=".2f", annot_kws={"size": 16})# font size
ax.set_ylim(5, 0)
plt.show()
           
