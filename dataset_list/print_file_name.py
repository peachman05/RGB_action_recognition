import os

def walk2(dirname):
    list_ = []
    for root, dirs, files in os.walk(dirname):
        # print(root)
        for filename in files:
            list_.append(os.path.join(root, filename))
    return list_


# path_dataset = 'F:\\Master Project\\Dataset\\sit_stand\\'
path_dataset = 'F:\\Master Project\\Dataset\\BUPT-dataset\\RGBdataset\\'
# path_dataset = 'F:\\Master Project\\Dataset\\KARD-split\\'

# list_file = walk2(path_dataset)
# action_list = ['run','sit','stand','standup','walk']
# action_list = ['run','sit','stand','walk','standup']
action_list = ['run','sit','stand','walk','standup']
# action_list = ['a01','a02','a03','a04','a13','a14']
# action_list = ['a01','a02','a03','a04','a05','a06','a07','a08','a09',
#                'a10','a11','a12','a13','a14','a15','a16','a17','a18']
action_id = {}
for i, name in enumerate(action_list):
    action_id[name] = i
# action_list = ['sit','stand','standup','sitdown']

# list_file = [list_file[0]]
# print(len(list_file))

list_file = walk2(path_dataset)
for file_path in list_file:
    name, extension = file_path.split('.') # [0] is path/filename, [1] is extension
    name_sp = name.split('\\')
    name_file = name_sp[-1]
    name_folder = name_sp[-2]
    name_bigsub = name_sp[-3]
    name_sum =  name_bigsub+'/'+name_folder+'/'+name_file
    if extension == 'mp4':
        print(name_sum, action_id[name_folder] )
