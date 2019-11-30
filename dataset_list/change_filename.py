import os

def walk2(dirname):
    list_ = []
    for root, dirs, files in os.walk(dirname):
        for filename in files:
            list_.append(os.path.join(root, filename))
    return list_

set_number = 2

path_dataset = 'F:\\Master Project\\Dataset\\BUPT-dataset\\RGBdataset0'+str(set_number)+'\\'

list_file = walk2(path_dataset)
# list_file = [list_file[0]]
print(len(list_file))
for file_path in list_file:
    split_s = file_path.split('.') # [0] is path/filename, [1] is extension
    new_name = split_s[0]+'_0'+str(set_number)+'.'+split_s[1] # filename_01.mp4
    os.rename(file_path, new_name)