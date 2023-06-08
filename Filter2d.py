import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    kernel_horisontal = np.array([[1, 2, 1],
                                  [0, 0, 0],
                                  [-1, -2, -1]])
    kernel_vertical = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_filtered = cv2.filter2D(frame, -1, kernel_horisontal) + cv2.filter2D(frame, -1, kernel_vertical)

    cv2.imshow("frame", frame_gray)
    cv2.imshow("frame_filtered", frame_filtered)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



