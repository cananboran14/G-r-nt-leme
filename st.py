import cv2
import numpy as np

cap = cv2.VideoCapture(0)
angle = 0
x_shift = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    (h, w) = frame.shape[:2]

    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        angle += 5
    elif key == ord('a'):
        angle -= 5
    elif key == ord('w'):
        x_shift += 10
    elif key == ord('q'):
        break

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += x_shift

    rotated = cv2.warpAffine(frame, M, (w, h))
    cv2.imshow('Klavye ile Manuel Kontrol', rotated)

cap.release()
cv2.destroyAllWindows()
