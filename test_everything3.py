import cv2
import numpy as np
import time
from keras.preprocessing import image


def get_sampling_frame( len_frames, path_video):   
  '''
  Sampling n_sequence frame from video file
  Input: 
      len_frames -- number of frames that this video have
  Output: 
      index_sampling -- n_sequence frame indexs from sampling algorithm 
  '''
  n_sequence = 10
  dim = (120,120)

  if True:#type_gen =='train':
      random_sample_range = 10
      if random_sample_range*n_sequence > len_frames:
          random_sample_range = len_frames//n_sequence

      if random_sample_range <= 0:
          print('test:',random_sample_range, len_frames, path_video)
      # Randomly choose sample interval and start frame
      if random_sample_range < 3:
          sample_interval = np.random.randint(1, random_sample_range + 1)
      else:
          sample_interval = np.random.randint(3, random_sample_range + 1)

      start_i = np.random.randint(0, len_frames - sample_interval * n_sequence + 1)
  
  # Get n_sequence index of frames
  index_sampling = []
  end_i = sample_interval * n_sequence + start_i
  for i in range(start_i, end_i, sample_interval):
      if len(index_sampling) < n_sequence:
          index_sampling.append(i)
  
  return index_sampling



path_video = "C:/Users/peachman/Desktop/jogging.mp4"
path_image = ["C:/Users/peachman/Desktop/sky1.jpg",
              "C:/Users/peachman/Desktop/sky2.jpg"]

start_time = time.time()
cap = cv2.VideoCapture(path_video)    
print('open video:', time.time() - start_time) # 0.01

start_time = time.time()
# image = cv2.imread(path_image) 
img = image.load_img(path_image[0])
print('open image:', time.time() - start_time) # 0.04

length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get how many frames this video have ,-1 because some bug          
    
index_sampling = get_sampling_frame(length_file, path_video) # get sampling index  


for j, n_pic in enumerate(index_sampling):
  start_time = time.time()
  cap.set(cv2.CAP_PROP_POS_FRAMES, n_pic) # jump to that index
  # print(time.time() - start_time) # set avg 0.012  
  ret, frame = cap.read()
  # print(time.time() - start_time) # 0.0015  , avg all = 0.01-0.016

  # start_time = time.time()
  # image = cv2.imread(path_image[j%2]) # 0.04
  # img = image.load_img(path_image[j%2]) # 0.00099
  print(time.time() - start_time) 
    # frame = get_crop_img(frame)
    # new_image = cv2.resize(frame, dim)
    # new_image = frame
    # new_image = new_image/255.0                
    # X1[i,j,:,:,:] = new_image