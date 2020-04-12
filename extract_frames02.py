import cv2
import glob
import os
import json

# path_dataset = 'F:/Master Project/Dataset/UCF11/'
path_dataset = 'F:\\Master Project\\Dataset\\UCF11'
extention_file = '.avi'

video_paths = glob.glob(os.path.join(path_dataset,"*", "*", "*"+extention_file))
all_file = len(video_paths)
print(all_file)

file = open('UCF11_len.txt', 'r')
len_dict = json.load(file)

for i, video_path in enumerate(video_paths):
    main_folder, sub_folder, name_file = video_path.split("\\")[-3:]  # basketball\v_shooting_01\v_shooting_01_01.avi
    name_file = name_file.split(extention_file)[0]
    full_folder_path = os.path.join(f"{path_dataset}-frames", main_folder, sub_folder, name_file)

    if os.path.exists(full_folder_path):
            continue

    os.makedirs(full_folder_path, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    success = True
    
    while success:
        final_path = (full_folder_path+"\\frame{:}.jpg").format(count)
        # final_path = 'F:\\test test\\image.jpg'
        cv2.imwrite(final_path, image)     # save frame as JPEG file
        success,image = vidcap.read()
        #   print('Read a new frame: ', success)
        count += 1
    len_dict[name_file] = count
    print("{:}/{:}".format(i,all_file), end="\r", flush=True)

with open('UCF11_len.txt', 'w') as file:
    json.dump(len_dict, file)


