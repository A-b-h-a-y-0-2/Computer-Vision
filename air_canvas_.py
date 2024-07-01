import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

# Brush settings
brush_thickness = 5
eraser_thickness = 20
mode = 'draw'  # can be 'draw' or 'erase'

# Giving different arrays to handle color points of different colors
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
all_points = [bpoints, gpoints, rpoints, ypoints]

# These indexes will be used to mark the points in particular arrays of specific color
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# The kernel to be used for dilation purpose 
kernel = np.ones((5,5),np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0
# Here is code for Canvas setup
paintWindow = np.zeros((471,780,3),dtype=np.uint8) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), (255,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), (0,255,0), 2)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), (0,0,255), 2)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), (0,255,255), 2)
paintWindow = cv2.rectangle(paintWindow, (620,1), (715,65), (0,0,0), 2)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "ERASER", (630, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True

# Slider callback function
def update_brush_thickness(x):
    global brush_thickness
    brush_thickness = x

# Create a window for the slider
cv2.namedWindow('Controls')
cv2.createTrackbar('Brush Thickness', 'Controls', brush_thickness, 20, update_brush_thickness)

while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (40,1), (110,65), (0,0,0), -1)
    frame = cv2.rectangle(frame, (120,1), (190,65), (255,0,0), -1)
    frame = cv2.rectangle(frame, (200,1), (270,65), (0,255,0), -1)
    frame = cv2.rectangle(frame, (280,1), (350,65), (0,0,255), -1)
    frame = cv2.rectangle(frame, (360,1), (430,65), (0,255,255), -1)
    frame = cv2.rectangle(frame, (440,1), (510,65), (0,0,0), -1)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "ERASER", (449, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0],landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0],landmarks[4][1])
        cv2.circle(frame, center, 3, (0,255,0),-1)
        
        if (thumb[1]-center[1]<30):
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 110: # Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:,:,:] = 255
            elif 120 <= center[0] <= 190:
                colorIndex = 0 # Blue
                mode = 'draw'
            elif 200 <= center[0] <= 270:
                colorIndex = 1 # Green
                mode = 'draw'
            elif 280 <= center[0] <= 350:
                colorIndex = 2 # Red
                mode = 'draw'
            elif 360 <= center[0] <= 430:
                colorIndex = 3 # Yellow
                mode = 'draw'
            elif 440 <= center[0] <= 510:
                mode = 'erase'
        else:
            if mode == 'draw':
                if colorIndex == 0:
                    bpoints[blue_index].appendleft((center, brush_thickness))
                elif colorIndex == 1:
                    gpoints[green_index].appendleft((center, brush_thickness))
                elif colorIndex == 2:
                    rpoints[red_index].appendleft((center, brush_thickness))
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft((center, brush_thickness))
            elif mode == 'erase':
                for points in all_points:
                    for point in points:
                        for i in range(len(point)):
                            if point[i] is not None:
                                if abs(point[i][0][0] - center[0]) < eraser_thickness and abs(point[i][0][1] - center[1]) < eraser_thickness:
                        # Draw a white line over the points to erase them in paintWindow
                                    cv2.line(paintWindow, point[i][0], point[i][0], (255, 255, 255), point[i][1])
                                    point[i] = None  # Mark the point as erased


    points = [bpoints, gpoints, rpoints, ypoints]

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1][0], points[i][j][k][0], colors[i], points[i][j][k][1])
                cv2.line(paintWindow, points[i][j][k - 1][0], points[i][j][k][0], colors[i], points[i][j][k][1])


    cv2.imshow("Output", frame) 
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):  # Save the canvas as an image
        cv2.imwrite('canvas.png', paintWindow)

cap.release()
cv2.destroyAllWindows()
