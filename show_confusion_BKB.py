from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from data_gen_bkb import DataGeneratorBKB
from model_ML import create_model_pretrain
from data_helper import readfile_to_dict

dim = (224,224)
n_sequence = 20
n_channels = 3
n_output = 4

path_dataset = 'F:\\Master Project\\Dataset\\BasketBall-RGB\\'

params = {'dim': dim,
          'batch_size': 1,
        #   'n_classes': 6,
          'n_sequence': n_sequence,
          'n_channels': n_channels,
          'path_dataset': path_dataset,
          'shuffle': False}

## dataset
test_txt = "dataset_list/trainlistBKB.txt"
test_d = readfile_to_dict(test_txt)
labels = test_d.copy()
num_mul = 50
partition = {'validation': list(test_d.keys()) * num_mul } # IDs
validation_generator = DataGeneratorBKB(partition['validation'] , labels, **params, type_gen='test')
predict_generator = DataGeneratorBKB(partition['validation'] , labels, **params, type_gen='predict')


weights_path = 'mobileNetV2-BKB-3ds-48-0.55.hdf5' # 15 frame
model = create_model_pretrain(dim, n_sequence, n_channels, n_output)
model.load_weights(weights_path)


## evaluate
# loss, acc = model.evaluate_generator(validation_generator, verbose=0)
# print(loss,acc)

# #### Confusion Matrix
y_pred_prob = model.predict_generator(predict_generator)
test_y = np.array(list(test_d.values()) * num_mul)
print("-----------")
print(y_pred_prob.shape)
print(len(test_y))

y_pred = np.argmax(y_pred_prob, axis=1)
normalize = True
cm = confusion_matrix(test_y, y_pred)
if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
else:
    print('Confusion matrix, without normalization')
  
accuracy = (cm[0,0] + cm[1,1] + cm[2,2] + cm[3,3] ) / 4
print("accuracy:",accuracy)

# classes = ['ApplyEyeMakeup','Archery','BabyCrawling','Basketball']
classes = ['0','1','2','3']

df_cm = pd.DataFrame(cm, columns=classes, index=classes)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
fig, ax = plt.subplots(figsize=(5,5))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,fmt=".2f", annot_kws={"size": 16})# font size
ax.set_ylim(5, 0)
plt.show()
