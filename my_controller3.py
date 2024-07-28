#EyeDrive

from vehicle import Driver
import numpy as np
import math
import time
import cv2
from collections import deque, Counter

driver = Driver()
sim_time = 0
timestep = driver.getBasicTimeStep()
headlights = driver.getDevice("headlights")
backlights = driver.getDevice("backlights")
# speed refers to the speed in km/h at which we want Altino to travel
speed = 0
# angle refers to the angle (from straight ahead) that the wheels
# currently have
angle = 0

# This the Altino's maximum speed
# all Altino controllers should use this maximum value
maxSpeed = 1.8
# ensure 0 starting speed and wheel angle
driver.setSteeringAngle(angle)
driver.setCruisingSpeed(speed)
# defaults for this controller
headlightsOn = False

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_faces(img):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame

def detect_eyes(img):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_frame, 1.2, 5)
    width = np.size(img, 1)
    height = np.size(img, 0)
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        eyecenter = x + w / 2
        if eyecenter < width / 2:
            margin_x = int(w * 0.15)  # Adjust the margin as needed
            margin_y = int(h * 0.1)
            left_eye = img[y + margin_y:y + h - margin_y, x + margin_x:x + w - margin_x]
        else:
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.1)
            right_eye = img[y + margin_y:y + h - margin_y, x + margin_x:x + w - margin_x]
    return left_eye, right_eye

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]
    return img

def blob_process(img, detector, threshold):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)  # 1
    img = cv2.dilate(img, None, iterations=4)  # 2
    img = cv2.medianBlur(img, 5)  # 3
    keypoints = detector.detect(img)
    return keypoints

def process_eye(eye, detector, threshold):
    if eye is not None:
        eye = cut_eyebrows(eye)
        keypoints = blob_process(eye, detector, threshold)
        eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return eye, keypoints
    return None, None

def analyze_gaze(keypoints, eye):
    if len(keypoints) == 0:
        return "No pupil detected"
    
    keypoint = keypoints[0]
    x = keypoint.pt[0]
    y = keypoint.pt[1]
    width = eye.shape[1]
    height = eye.shape[0]
    
    if x < 0.52 * width:
        horizontal = "Right"
    elif x > 0.6 * width:
        horizontal = "Left"
    else:
        horizontal = "Center"
    
    return f"{horizontal}"

def nothing(x):
    pass

def get_most_common_gaze(gaze_list):
    if len(gaze_list) == 0:
        return "No gaze data"
    gaze_counts = Counter(gaze_list)
    most_common_gaze = gaze_counts.most_common(1)[0][0]
    return most_common_gaze

cap = cv2.VideoCapture(0)
cv2.namedWindow('my image')
cv2.createTrackbar('threshold', 'my image', 0, 255, nothing)
if not cap.isOpened():
    print("Camera is disconnected")

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1600  # To prevent false pupil detection (this is a very large area which a pupil would never cover)
detector = cv2.SimpleBlobDetector_create(detector_params)
gaze_history = deque(maxlen=30)  # Store gaze history for the last 1 seconds (30 FPS)
checker_history = deque(maxlen=5)
checker_history.append(1)
checker_history.append(1)
move=0
eye_status=""
cen=1
i=1
firstclosetimer=0
#MAIN MAIN MAIN MAIN
checker=0
while driver.step() != -1:
        
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture frame from camera")
        break

    face_frame = detect_faces(frame)
    if face_frame is not None:
        eyes = detect_eyes(face_frame)
        threshold = cv2.getTrackbarPos('threshold', 'my image')
        nothing(threshold)
        left_eye, left_keypoints = process_eye(eyes[0], detector, threshold)
        right_eye, right_keypoints = process_eye(eyes[1], detector, threshold)

        left_gaze = "No left eye detected"
        if left_eye is not None:
            left_gaze = analyze_gaze(left_keypoints, left_eye)
            cv2.imshow('left eye', left_eye)
            eye_status = "Eyes open"
        else:
            eye_status = "Eyes closed"

        right_gaze = "No right eye detected"
        if right_eye is not None:
            right_gaze = analyze_gaze(right_keypoints, right_eye)
            cv2.imshow('right eye', right_eye)

        if right_eye is None and left_eye is None:
            eye_status = "Eyes closed"
        else:
            eye_status = "Eyes open"
            
            
        gaze_history.append(left_gaze)
        most_common_gaze = get_most_common_gaze(gaze_history)
        turn=most_common_gaze
        cv2.putText(frame, f"Gaze: {left_gaze}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Most Common Gaze (1s): {most_common_gaze}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Turn: {turn}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    

        
    
    if(eye_status=="Eyes closed"):
        move=0
        
    if(turn.startswith("No")):
        move=0
        checker=1
    else:
        checker=0
        
    if(checker_history[-1] != checker):
        if(checker_history[-1]==0 and checker==1):
            if(i==1):
                i=0
            elif(i==0):
                i=1    
        checker_history.append(checker)

        
    if(i==0): #Check first eye opening(And alternate eye openings)
        move=1 #Move every step by Default 
    else:
        move=0
    
    
        
    
    print("\nmove = ",move,"\nturn = ", turn,"\ntype_turn = ", type(turn),\
         "\neye_status = ",eye_status,\
         "\nEyes Went from closed to open prev? ",i,
         "\nChecker History = ",checker_history)
         
    if(move == 1):
    
        if turn == "Left":
            speed=0.9
            angle=-0.2
            cen=0
        if turn == "Right":
            speed=0.9
            angle=0.2
            cen=0
    
        if turn == "Center":
            speed=4.5
            angle=0
            cen=1
    
        if cen == 0:
            print("Turning")
        else:       
            print("Straight")
    else:
        speed=0
        angle=0
                    
    cv2.imshow('my image', frame)
    sim_time += timestep * 0.001
    print("simulation timer: ", sim_time)
    print("speed", speed)
    
    if sim_time >= 300:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    if speed >= 1.8:
        speed=1.8
        
    if angle>=0.4:
        angle=0.4
    

    
    driver.setCruisingSpeed(speed)
    driver.setSteeringAngle(angle)
