from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from data_gen_bkb import DataGeneratorBKB 
from model_ML import create_model_pretrain
from data_helper import readfile_to_dict

dim = (224,224)
n_sequence = 6
n_channels = 3
n_output = 5
# base_path = 'D:\\Peach\\'
base_path = 'F:\\Master Project\\'
# path_dataset = base_path+'Dataset\\sit_stand\\'
# path_dataset = base_path + 'Dataset\\BUPT-dataset\\RGBdataset_crop\\'
path_dataset = base_path + 'Dataset\\sit_stand_crop02\\'

params = {'dim': dim,
          'batch_size': 1,
        #   'n_classes': 6,
          'n_sequence': n_sequence,
          'n_channels': n_channels,
          'path_dataset': path_dataset,
        #   'option': 'RGBdiff',
          'shuffle': False}

## dataset
test_txt = "dataset_list/testlistBUPT_crop02.txt"
test_d = readfile_to_dict(test_txt)
labels = test_d.copy()
num_mul = 1
print(len(test_d.keys()))
key_list = list(test_d.keys()) * num_mul
partition = {'validation': key_list  } # IDs
# validation_generator = DataGeneratorBKB(key_list , labels, **params, type_gen='test')
predict_generator = DataGeneratorBKB(key_list , labels, **params, type_gen='predict')


# weights_path = 'BUPT-2d-equalsplit-RGBdif-72-0.98-0.90.hdf5' 
# weights_path = 'BUPT-RGB-Crop-96-0.92-0.88.hdf5' 
# weights_path = 'BUPT-RGB-Crop-alpha-035-210-0.93-0.92.hdf5' 
weights_path = 'BUPT-RGB-Crop-alpha-addDTS05-129-0.91-0.70.hdf5'
# weights_path = 'BUPT-RGB-Crop-alpha-addDTS03-300-0.97-0.63.hdf5'

model = create_model_pretrain(dim, n_sequence, n_channels, n_output, 'MobileNetV2')
model.load_weights(weights_path)


## evaluate
# loss, acc = model.evaluate_generator(validation_generator, verbose=0, workers=0)
# print(loss,acc)

# #### Confusion Matr
y_pred_prob = model.predict_generator(predict_generator, workers=0)
test_y = np.array(list(test_d.values()) * num_mul)
print("-----------")
print(y_pred_prob.shape)
print(len(test_y))
 
y_pred = np.argmax(y_pred_prob, axis=1)
normalize = True

all_y = len(test_y)
sum = all_y
for i in range(len(y_pred)):
    if test_y[i] != y_pred[i]:
        sum -= 1
        print(key_list[i],' actual:',test_y[i],'predict:',y_pred[i])

cm = confusion_matrix(test_y, y_pred)
if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
else:
    print('Confusion matrix, without normalization')


accuracy = sum / all_y
# # accuracy = (cm[0,0] + cm[1,1]) / 4
print("accuracy:",accuracy)

print(cm)

# classes = ['ApplyEyeMakeup','Archery','BabyCrawling','Basketball']
# classes = ['0','1','2','3']
# classes = ['sit','stand','standup']
# classes = ['run','walk','stand']
classes = ['run','sit','stand','walk', 'standup']
# classes = ['run','sit','stand','walk']

df_cm = pd.DataFrame(cm, columns=classes, index=classes)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
fig, ax = plt.subplots(figsize=(5,5))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,fmt=".2f", annot_kws={"size": 16})# font size
ax.set_ylim(5, 0)
plt.show()
