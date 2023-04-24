import math
import cv2
import mediapipe as mp
from time import time
    
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode = False,
    min_detection_confidence = 0.5,
    model_complexity=0
    )
# de al library al feha al pose detection model
mp_drawing = mp.solutions.drawing_utils

pose_timer_start = 0
timer = 3
labels = {"unknowen":0,"Close Arms":0,"Crotch Arms":0,"Square Arms":0,"incomplete":0}
label = ""


if __name__ != "__main__":

    def getPose(frame, pose):
        """given frame and pose, returns the landmarks in a list and draws them on the frame

        Args:
            frame (ndarray): numpy array with our image pixel values 
            pose (mediapipe model): this is the model that does the pose detection

        Returns:
            list: x,y,z quadrants of the different body points in one list
        """
        
        # convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #  get the landmarks
        results = pose.process(frame_rgb)
        
        height, width, _ = frame.shape
        landmarks = []
        
        if results.pose_landmarks:
            
            # draw the landmarks on our frame
            mp_drawing.draw_landmarks(image = frame,
                                    landmark_list = results.pose_landmarks,
                                    connections = mp_pose.POSE_CONNECTIONS)
            
            # append the points to the list for furthur use
            for landmark in results.pose_landmarks.landmark:
                
                landmarks.append((int(landmark.x*width), 
                                int(landmark.y*height),
                                (landmark.z * width)))


        return landmarks

    def get_angle(p1,p2,p3):
        """function that gets the inner angle between three points

        Args:
            p1 (tuple 3x1): tuple that holds the quadrants of the point
            p2 (tuple 3x1): tuple that holds the quadrants of the point
            p3 (tuple 3x1): tuple that holds the quadrants of the point

        Returns:
            int: inner angle, which is between 0 - 360
        """
        # unpack the tuple
        # the z quadrant is unimportant
        x1,y1,_ = p1
        x2,y2,_ = p2
        x3,y3,_ = p3
        
        angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2) )
        
        if angle <0:
            
            angle += 360
            
        return angle

    def get_distance(p1,p2):
        """function that returns the distance between two 3d points

        Args:
            p1 (tuple): 1st point x,y,z coordinates
            p2 (tuple): 2st point x,y,z coordinates

        Returns:
            int: the Euclidean distance between these two points
        """
        x1,y1,_ = p1
        x2,y2,_ = p2
        
        distance = math.sqrt(  ((x2-x1)**2) + ((y2-y1)**2) )
        
        return distance

    def classify(landmarks):

        # by default label is unknowen
        label = "unknowen"
    
        # in case not all points are found
        if(len(landmarks) < 20):
            label = "incomplete"
            return  label
        
        
        # bicep angle
        left_elbow_angle = get_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
        
        right_elbow_angle = get_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        
        
        # armpit angel
        left_armpit_angle = get_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        
        right_armpit_angle = get_angle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
        
        
        # distances
        # wrist to elbow
        rightwrist_leftelbow_dist= get_distance(
                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                                    )
        rightwrist_rightelbow_dist= get_distance(
                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                                    )
        leftwrist_leftelbow_dist= get_distance(
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                                    )
        leftwrist_rightelbow_dist= get_distance(
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                                    )
        
        # wrist to hip
        rightwrist_righthip_dist= get_distance(
                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                                    )
        leftwrist_lefthip_dist= get_distance(
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                                    )
        
        # wrist to wrist
        rightwrist_leftwrist_dist= get_distance(
                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                                    )
        
        # hip to hip
        righthip_lefthip_dist= get_distance(
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                                    )
        
        
        # test square arms
        if(right_elbow_angle >50 and right_elbow_angle <110):
            if(left_elbow_angle > 230 and left_elbow_angle < 300):
                if(right_armpit_angle >320 or right_armpit_angle <20):
                    if(rightwrist_leftelbow_dist<rightwrist_rightelbow_dist):
                        if(leftwrist_rightelbow_dist<leftwrist_leftelbow_dist):
                            if(left_armpit_angle > 320 or left_armpit_angle < 20):
                                label = "Square Arms"
        
        
        # test close to body arms
        if(rightwrist_righthip_dist < rightwrist_rightelbow_dist):
            if(leftwrist_lefthip_dist < leftwrist_leftelbow_dist):
                if(rightwrist_leftwrist_dist < righthip_lefthip_dist):  
                    label = "Crotch Arms"
                else:
                    label = "Close Arms"
                    
        return label
    
    
    def detect_pose(frame):
        global pose
        global pose_timer_start 
        global labels
        global label 

        landmarks = getPose(frame, pose)

        cur_label  = classify(landmarks)

        pose_timer_end = time()

        if(pose_timer_end - pose_timer_start >= 1):

            label = max(labels, key=labels.get)
            labels = {"unknowen":0,"Close Arms":0,"Crotch Arms":0,"Square Arms":0,"incomplete":0}
            pose_timer_start = pose_timer_end
            
        labels[cur_label] +=1
        
        if label == "unknowen":
            color  = (0,0,255)
        elif label == "incomplete":
            color = (255,0,0)
        else:
            color = (0,255,0)


        cv2.putText(frame,
            label,
            (50,50),
            cv2.FONT_HERSHEY_PLAIN, 2,
            color, 3)
        
        return label
        