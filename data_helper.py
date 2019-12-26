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