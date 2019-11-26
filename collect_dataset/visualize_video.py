import cv2

# Parameter
path_dataset = 'F:\\Master Project\\Dataset\\BasketBall-RGB\\'
action_select = 3 # 0=dribble, 1=shoot, 2=pass, 3=stand
number_file = '25'

action_list = ['dribble','shoot','pass','stand']
action = action_list[action_select]
path_save = path_dataset +'\\'+action+'\\'+action
path_file = path_save + number_file + '.mp4'
print(path_file)
cap = cv2.VideoCapture(path_file)
length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # get length of frames
print('length_file:',length_file)