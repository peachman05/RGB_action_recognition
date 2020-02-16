import cv2

def readfile_to_dict(filename):
    d = {}
    f = open(filename)
    for line in f:
        # print(str(line))
        if line != '\n':
            (key, val) = line.split()
            d[key] = int(val)

    return d

def calculateRGBdiff(sequence_img, stop_frame):
    'keep first frame as rgb data, other is use RGBdiff for temporal data'
    length = len(sequence_img)        
    # find RGBdiff frame 2nd to last frame
    for i in range(length-1,stop_frame,-1): # count down
        sequence_img[i] = cv2.subtract(sequence_img[i],sequence_img[i-1])        

    return sequence_img

def calculate_intersection(cordinate1, cordinate2): 
    # print(cordinate1,cordinate2)
    x_l1, y_l1, x_r1, y_r1 = cordinate1
    x_l2, y_l2, x_r2, y_r2 = cordinate2
    # itc = intersect
    itc_x_rp = min(x_r1,x_r2)
    itc_x_lp = max(x_l1,x_l2)
    itc_y_rp = min(y_r1,y_r2)
    itc_y_lp = max(y_l1,y_l2)
    if itc_x_rp < itc_x_lp:
        return 0 # not intersect
    else:
        itc_x_length = itc_x_rp - itc_x_lp
        itc_y_length = itc_y_rp - itc_y_lp
        return itc_x_length*itc_y_length