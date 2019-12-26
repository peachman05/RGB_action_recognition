import cv2

path_dataset = 'F:/Master Project/Dataset/sit_stand/'
action_list = ['run','walk']
action_select = 0
action = action_list[action_select]
n_point_view = 5
n_repeat = 5
split_size = 4
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
width = 640
height = 480
for p_i in range(1,n_point_view):
    for r_i in range(n_repeat):        
        path_read = '{:}/{:}/{:}{:02d}_{:02d}.mp4'.format(path_dataset,action,action,p_i,r_i)
        cap = cv2.VideoCapture(path_read)
        length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_per_split = length_file//split_size

        for s_i in range(split_size):
            path_write = '{:}/{:}_split/{:}{:02d}_{:02d}_{:02d}.mp4'.format(path_dataset,action,action,p_i,r_i,s_i)            
            out = cv2.VideoWriter(path_write, fourcc, 30.0, (width, height))
            for i in range(frame_per_split):
                ret, frame = cap.read()
                out.write(frame)
            out.release()
        cap.release()