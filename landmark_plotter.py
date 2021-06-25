import cv2
import sleep_detector

# Start the webcam
webcam = cv2.VideoCapture(0)

# Capture images in a loop
frame_present, frame = webcam.read()
while frame_present and not(cv2.waitKey(15) & 0xFF == ord('d')):
    frame_present, frame = webcam.read()
    sleep_detector.plot_landmarks(frame)
    cv2.imshow('Landmarks plotted', frame)

webcam.release()
cv2.destroyAllWindows()




