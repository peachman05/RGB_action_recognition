from imageai.Detection import ObjectDetection
import os
import tensorflow as tf
import cv2

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

execution_path = os.getcwd()

detector = ObjectDetection()
# detector.setModelTypeAsTinyYOLOv3()
detector.setModelTypeAsYOLOv3()
# detector.setModelTypeAsRetinaNet() 
# detector.setModelPath( os.path.join(execution_path , "yolo-tiny.h5"))
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
# detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel(detection_speed="fast")#detection_speed="fast"

action = 'standup'
path_file = 'F:\\Master Project\\Dataset\\sit_stand\\'+action+'\\'+action+'00_02.mp4'
cap = cv2.VideoCapture(path_file)
dim = (224,224)
for i in range(80):
    ret, frame = cap.read()

    detect_image, detections, extract_picture = detector.detectObjectsFromImage(input_type="array", input_image=frame, output_type='array', 
                                                 minimum_percentage_probability=10, extract_detected_objects = True )
    print('round:',i)

    max_prob = 0
    max_idx = 0
    for i,eachObject in enumerate(detections):
        if eachObject["name"] == 'person' and eachObject["percentage_probability"] > max_prob:
            max_prob = eachObject["percentage_probability"]
            max_idx = i
            print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ",
              eachObject["box_points"] )
            print("--------------------------------")
    # for eachObject in detections:'
    crop_img = extract_picture[max_idx]
    new_image = cv2.resize(crop_img, dim)
    # new_image = new_image/255.0
    
    cv2.imshow('Frame',new_image)
    cv2.waitKey(15)

print("finish!")