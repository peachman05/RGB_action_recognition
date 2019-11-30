import os

def walk2(dirname):
    list_ = []
    for root, dirs, files in os.walk(dirname):
        print(root)
        for filename in files:
            list_.append(os.path.join(root, filename))
    return list_


path_dataset = 'F:\\Master Project\\Dataset\\BUPT-dataset\\RGBdataset\\'

list_file = walk2(path_dataset)
action_list = ['run','sit','stand','standup','walk']

# list_file = [list_file[0]]
print(len(list_file))
for file_path in list_file:
    name, extension = file_path.split('.') # [0] is path/filename, [1] is extension
    name_sp = name.split('\\')
    name_file = name_sp[-1]
    name_folder = name_sp[-2]
    
    if extension == 'mp4':
       print(name_folder+'\\'+name_file )
    # elif extension == 'npy':
    #     data = np.load(file_path)
    #     print(file_path,'npy:',data.shape[0])
    