import cv2
import numpy as np

cap = cv2.VideoCapture(0)
previous_frame = None

while True:
    _, frame = cap.read()

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5)
    final_img = clahe.apply(frameGray) + 30

    cv2.imshow("frame", frame)
    cv2.imshow("final_img", final_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()