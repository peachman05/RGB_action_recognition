import numpy as np

path_dataset = 'F:\\Master Project\\Dataset\\BasketBall-RGB\\'
action_select = 3 # 0=dribble, 1=shoot, 2=pass, 3=stand
number_file = '30'

action_list = ['dribble','shoot','pass','stand']
action = action_list[action_select]
path_save = path_dataset +'\\'+action+'\\'+action
path_file = path_save + number_file + '.npy'
print(path_file)
skeleton_data = np.load(path_file)

print(skeleton_data.shape)
index_sampling = [0,1,2]
skeleton_data = skeleton_data[index_sampling]
skeleton_data = skeleton_data[:,:,[0,1]]
skeleton_data = skeleton_data.reshape(3,3*2)
print(skeleton_data.shape)
