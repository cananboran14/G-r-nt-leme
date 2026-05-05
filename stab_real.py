import cv2
import numpy as np
import time

# --- PARAMETRELER ---
ALPHA = 0.03          # Küçükse daha stabil, büyükse daha hızlı tepki verir
ZOOM = 1.08           # Siyah kenarları azaltmak için hafif zoom
MAX_CORNERS = 150
QUALITY_LEVEL = 0.03
MIN_DISTANCE = 30


def warp_affine_transform(frame, dx, dy, da, zoom=1.08):
    h, w = frame.shape[:2]
    center = (w / 2, h / 2)

    # Rotasyonu merkeze göre uygula
    M = cv2.getRotationMatrix2D(center, np.degrees(da), zoom)

    # Stabilizasyon kaydırmasını ekle
    M[0, 2] += dx
    M[1, 2] += dy

    stabilized = cv2.warpAffine(
        frame,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return stabilized


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Hata: Kamera açılamadı.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, prev_frame = cap.read()

if not ret:
    print("Hata: İlk frame alınamadı.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Trajectory değişkenleri
cur_x, cur_y, cur_a = 0.0, 0.0, 0.0
sm_x, sm_y, sm_a = 0.0, 0.0, 0.0

prev_time = time.time()

while True:
    ret, curr_frame = cap.read()

    if not ret:
        print("Frame alınamadı.")
        break

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # 1. Feature Detection
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=7
    )

    dx, dy, da = 0.0, 0.0, 0.0

    if prev_pts is not None and len(prev_pts) > 10:
        # 2. Optical Flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            prev_pts,
            None
        )

        if curr_pts is not None and status is not None:
            idx = np.where(status.flatten() == 1)[0]

            if len(idx) > 10:
                p1 = prev_pts[idx]
                p2 = curr_pts[idx]

                # 3. Affine hareket tahmini
                m, inliers = cv2.estimateAffinePartial2D(
                    p1,
                    p2,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3
                )

                if m is not None:
                    dx = m[0, 2]
                    dy = m[1, 2]
                    da = np.arctan2(m[1, 0], m[0, 0])

    # 4. Kamera hareketini trajectory olarak biriktir
    cur_x += dx
    cur_y += dy
    cur_a += da

    # 5. Low-pass filter
    sm_x = (1 - ALPHA) * sm_x + ALPHA * cur_x
    sm_y = (1 - ALPHA) * sm_y + ALPHA * cur_y
    sm_a = (1 - ALPHA) * sm_a + ALPHA * cur_a

    # 6. Düzeltme farkı
    diff_x = sm_x - cur_x
    diff_y = sm_y - cur_y
    diff_a = sm_a - cur_a

    stabilized_frame = warp_affine_transform(
        curr_frame,
        diff_x,
        diff_y,
        diff_a,
        zoom=ZOOM
    )

    # 7. FPS
    curr_time = time.time()
    fps = 1 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time

    # 8. Görselleştirme
    res = np.hstack((curr_frame, stabilized_frame))

    cv2.putText(
        res,
        f"Original",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        res,
        f"Stabilized",
        (curr_frame.shape[1] + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        res,
        f"FPS: {int(fps)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Real-Time Webcam Stabilization", res)

    prev_gray = curr_gray.copy()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
