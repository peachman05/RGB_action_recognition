import cv2
import time
import winsound
import os

action_select = 2
point_view = 4
run_time = 3 # second
round = 20
start = 0

path_dataset = 'F:\\Master Project\\Dataset\\sit_stand\\'
action_list = ['sit','stand','standup','sitdown']
action = action_list[action_select]
path_save = path_dataset +'\\'+action+'\\'+action+'{:02d}'.format(point_view)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
width = 640
height = 480

cap = cv2.VideoCapture(0)
for i in range(start,round):
    name_video = path_save+'_{:02d}'.format(i)+'.mp4'  
    if os.path.exists(name_video):
        print(name_video,' is exits!!!!')
        break
    out = cv2.VideoWriter(name_video, fourcc, 30.0, (width, height))
    # print('----round:',i,'-------')
    print(i,action_list[action_select])
    for i in range(5,0,-1):
        print("start in:", i)
        winsound.Beep(1000, 100)
        time.sleep(1) 

    start_time = time.time()
    count_frame = 0  
    while(cap.isOpened()):  
        ret, frame = cap.read()  
        out.write(frame)
        cv2.imshow('Frame',frame)        
        dif_t = (time.time() - start_time)
        count_frame += 1        
        cv2.waitKey(10)
        if dif_t > run_time:
            print('time out!!!!') 
            break
        print("time: {:02} s".format(int(dif_t)%60), end='\r')
        
    print('save:',name_video,'frames:',count_frame)
    out.release()



 

#     ret, frame = cap.read()  
#     cv2.imshow('Frame',frame)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#             break 

# print(frame.shape)

