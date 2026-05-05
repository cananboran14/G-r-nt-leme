import cv2
import numpy as np
import os

# --- AYARLAR ---
VIDEO_PATH = os.path.join(os.path.expanduser("~"), "Downloads", "stab_shake.mp4")
SMOOTHING_RADIUS = 50  # Bu değeri 20 ile 100 arasında deneyin. Artarsa sarsıntı azalır.
ZOOM = 1.15            # Kenar hatalarını gizlemek için.

cap = cv2.VideoCapture(VIDEO_PATH)

# Video özelliklerini al
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 1. ADIM: Tüm kareler arasındaki ham dönüşümleri (transform) çıkar
transforms = np.zeros((n_frames-1, 3), np.float32) 

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

print("Video analiz ediliyor, lütfen bekleyin...")

for i in range(n_frames - 1):
    ret, curr_frame = cap.read()
    if not ret: break

    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Köşeleri bul (Daha hassas ayar)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    
    # Optik akış
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # Sadece başarılı eşleşmeleri al
    idx = np.where(status == 1)[0]
    p1 = prev_pts[idx]
    p2 = curr_pts[idx]

    # Matris tahmini (Sadece rotasyon ve kayma)
    m, _ = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    
    if m is not None:
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
    else:
        dx = dy = da = 0

    transforms[i] = [dx, dy, da]
    prev_gray = curr_gray
    if i % 50 == 0: print(f"Kare {i}/{n_frames} analiz edildi.")

# 2. ADIM: Yörüngeyi (Trajectory) hesapla ve yumuşat
trajectory = np.cumsum(transforms, axis=0)

def moving_average(curve, radius):
    window_size = 2 * radius + 1
    # Kenarları uzat (Drifti önlemek için kritik nokta!)
    curve_pad = np.pad(curve, (radius, radius), mode='edge')
    # Hareketli ortalama filtresi
    return np.convolve(curve_pad, np.ones(window_size)/window_size, mode='valid')

smoothed_trajectory = np.zeros_like(trajectory)
for i in range(3):
    smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius=SMOOTHING_RADIUS)

# 3. ADIM: Yeni (Yumuşatılmış) dönüşümleri oluştur
# Farkı bul ve orijinal harekete ekle
diff = smoothed_trajectory - trajectory
transforms_smooth = transforms + diff

# 4. ADIM: Videoyu oku ve yeni dönüşümleri uygula
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
print("İşlenmiş video oynatılıyor...")

for i in range(n_frames - 1):
    ret, frame = cap.read()
    if not ret: break

    dx, dy, da = transforms_smooth[i]

    # Dönüşüm matrisini oluştur
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    # Uygula
    frame_stab = cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Zoom uygula (Kenardaki siyah titremeleri siler)
    if ZOOM != 1.0:
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, 0, ZOOM)
        frame_stab = cv2.warpAffine(frame_stab, matrix, (w, h))

    # Karşılaştırmalı göster
    combined = np.hstack((frame, frame_stab))
    cv2.imshow("Stabilization (L: Orijinal - R: Stabilize)", cv2.resize(combined, (w, h//2)))
    
    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
