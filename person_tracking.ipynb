import cv2
import numpy as np
from imageai.Detection import ObjectDetection
from centroidTracker import CentroidTracker

### ImageAI
model_path = "pretrain/yolo-tiny.h5"
# model_path = "pretrain/yolo.h5"
detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
# detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.loadModel(detection_speed="flash") #"normal"(default), "fast", "faster" , "fastest" and "flash".
custom_objects = detector.CustomObjects(person=True)

# tracking define
centroidTracker = CentroidTracker()

cap = cv2.VideoCapture("C:/Users/peachman/Desktop/jogging.mp4")
while(cap.isOpened()):
    ret, frame = cap.read()
    returned_image, detections, extract_image = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
                    input_type='array', input_image=frame, output_type='array',
                    minimum_percentage_probability=20,
                    extract_detected_objects = True )
    
    # tracking
    all_boxs = [detect_obj["box_points"] for detect_obj in detections] 
    
    centroidTracker.updateTrack(all_boxs)
    draw_frame = centroidTracker.drawTrack(frame)

    cv2.imshow('Frame', draw_frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break 