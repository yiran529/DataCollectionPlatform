import cv2

cap = cv2.VideoCapture(0)  # /dev/video0
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if cap.isOpened():
    print("Camera opened successfully!")
else:
    print("Failed to open camera.")
cap.release()