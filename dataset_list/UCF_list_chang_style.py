

def train_change(f_read_name, f_write_name):
    f_read = open(f_read_name, 'r')
    f_write = open(f_write_name , 'w')

    for line in f_read:
        if line != '\n':
            (key, val) = line.split()
            f_write.write("{:} {:}\n".format(key, int(val)-1))

    f_read.close()
    f_write.close()


def test_change(f_read_name, f_write_name, f_label_name):
    f_read = open(f_read_name, 'r')
    f_write = open(f_write_name , 'w')
    f_label = open(f_label_name, 'r')

    label_mapping = {}
    for line in f_label:
        if line != '\n':
            (index, class_name) = line.split()
            label_mapping[class_name] = int(index) - 1 
 
    for line in f_read:
        if line != '\n':
            video_name = line.split()[0] # delete new line
            folder_name = line.split('/')[0] # name of folder
            f_write.write("{:} {:}\n".format(video_name, label_mapping[folder_name]))

    f_read.close()
    f_write.close()


old_train_txt = "UCF-101/trainlist01.txt"
old_test_txt = "UCF-101/testlist01.txt"

new_train_txt = "UCF-101/trainlistUCF-101.txt"
new_test_txt = "UCF-101/testlistUCF-101.txt"
label_txt = "UCF-101/classInd.txt"

train_change(old_train_txt, new_train_txt)
test_change(old_test_txt, new_test_txt, label_txt)