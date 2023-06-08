import numpy as np
import cv2

cap = cv2.VideoCapture(0)
_, frame = cap.read()
prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame)
hsv[..., 1] = 255

while True:
    _, frame2 = cap.read()
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    prev_frame = next_frame

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    test_hsv = hsv.copy()
    test_hsv[..., 2] = np.where((hsv[..., 0] > 100), hsv[..., 2], 0)
    test_bgr = cv2.cvtColor(test_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('test_frame', test_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()