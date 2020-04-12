
import math 
import collections

INF = float("inf")

class Person:
    def __init__(self, cen_point, box):
        self.cen_point = [0.0, 0.0] # x, y
        self.box = [0.0, 0.0, 0.0, 0.0] # x1,x2,y1,y2
        # For comparison
        self.min_distance = INF
        self.min_new_id = -1

class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        self.people_dict = collections.OrderedDict()
        self.id_count = 0

    def calculateCentroid(self, all_boxs):
        centroids = []
        for obj in all_boxs:
            x1,y1,x2,y2 = obj
            x_cen = (x1+x2)/2
            y_cen = (y1+y2)/2
            centroids.append([x_cen,y_cen])    
        return centroids

    def calculateDistance(self, p1, p2):
        return math.sqrt(p1*p1 + p2*p2)

    def resetPreviousTemp(self):
        for id in self.people_dict:
            people_dict[id].min_distance = INF
            people_dict[id].min_new_id = -1

    def updateTrack(self, all_boxs):
        self.resetPreviousTemp()
        isNewObject = [True] * len(all_boxs) # assume all is new object
        centroids = self.calculateCentroid(all_boxs)
        for i,centroid in enumerate(centroids): # new detect person
            min_dis = INF
            min_id = -1
            for id in self.people_dict: # old register person and find min distance
                d = self.calculateDistance(centroid, self.people_dict[id].cen_point)
                if d < min_dis:
                    min_dis = d
                    min_id = id

            if min_id != -1 and min_dis < self.people_dict[min_id].min_distance: # lower than previous
                if min_new_id == -1: # not conflict
                    isNewObject[i] = False  # set isNewObject[new_id] from true to false(can find old object)
                else: # conflict with previous
                    old_setID = self.people_dict[min_id].min_new_id
                    isNewObject[old_setID] = True # set back to newObject
                self.people_dict[min_id].min_distance = d
                self.people_dict[min_id].min_new_id = i
                self.people_dict[min_id].box = all_boxs[i]
                self.people_dict[min_id].cen_point = centroids[i]

        print(isNewObject)

        for i, isNew in enumerate(isNewObject):
            if isNew:
                self.people_dict[self.id_count] = Person(centroids[i], all_boxs[i])
                self.id_count += 1        
    
    def drawTrack(self, frame):
        for id in self.people_dict:
            x1,y1,x2,y2 = self.people_dict[id].box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)                
            cv2.rectangle(frame,(x1-1,y1),(x2+1,y1+40),(0,255,0),-1) # draw background text
            cv2.putText(frame, str(id), (x1,y1+30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

                    

