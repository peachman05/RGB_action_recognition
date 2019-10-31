import cv2
import glob
import os

print(cv2.__version__)

# path_dataset = 'F:\\Master Project\\Dataset\\KARD-split'
# extention_file = '.mp4'

path_dataset = 'F:\\Master Project\\Dataset\\UCF-101-Temp'
extention_file = '.avi'

path_file = '/a01/a01_s01_e01.mp4'
# vidcap = cv2.VideoCapture(path_dataset + path_file)
# success,image = vidcap.read()
# count = 0
# success = True
# while success:
#   cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
#   success,image = vidcap.read()
#   print('Read a new frame: ', success)
#   count += 1

video_paths = glob.glob(os.path.join(path_dataset, "*", "*"+extention_file))
all_file = len(video_paths)
for i, video_path in enumerate(video_paths):
    sequence_type, sequence_name = video_path.split("\\")[-2:]  # a01\a01_s01_e01.mp42
    sequence_name = sequence_name.split(extention_file)[0]
    sequence_path = os.path.join(f"{path_dataset}-frames", sequence_type, sequence_name)

    if os.path.exists(sequence_path):
            continue

    os.makedirs(sequence_path, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    success = True
    
    while success:
        final_path = (sequence_path+"\\frame{:}.jpg").format(count)
        # final_path = 'F:\\test test\\image.jpg'
        cv2.imwrite(final_path, image)     # save frame as JPEG file
        success,image = vidcap.read()
        #   print('Read a new frame: ', success)
        count += 1

    print("{:}/{:}".format(i,all_file), end="\r", flush=True)