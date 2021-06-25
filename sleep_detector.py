import cv2
import dlib
import numpy as np
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

sleep = 0
drowsy = 0
active = 0
status = ''
color = (0, 0, 0)

def compute(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)

def eye_status(a, b, c, d, e, f):

    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking if it is blinked
    if ratio > 0.25:
        # Awake
        return 2
    elif 0.21 < ratio <= 0.25:
        # Drowsy
        return 1
    else:
        # Sleeping
        return 0

def plot_landmarks(frame):

    global sleep
    global drowsy
    global active
    global status
    global color

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        shape = predictor(gray, face)
        landmarks = face_utils.shape_to_np(shape)

        left_status = eye_status(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_status = eye_status(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if left_status == 0 or right_status == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = 'Sleeping'
                color = (255, 0, 0)

        elif left_status == 1 or right_status == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = 'Drowsy'
                color = (0, 0, 255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = 'Active'
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for landmark in landmarks:
            cv2.circle(frame, tuple(landmark), 1, (255, 255, 255), -1)

