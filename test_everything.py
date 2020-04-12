# from data_helper import calculateRGBdiff, calculate_intersection

# p1 = (160, 125, 550, 427)
# p2 = (448, 220, 539, 379)

# area = calculate_intersection(p1, p2)

# print(area)
from model_ML import create_model_pretrain, create_model_Conv3D

dim = (120,120) # เพิ่งเปลี่ยนจาก 120
n_sequence = 10 # เพิ่งเปลี่ยน จาก 10
n_channels = 3
n_output = 5
model_type = ''#'Conv3D'

if model_type == 'Conv3D':
    model = create_model_Conv3D(dim, n_sequence, n_channels, n_output, set_pretrain=False)    
else:
    model = create_model_pretrain(dim, n_sequence, n_channels, n_output, 0.35)

print(model.summary())