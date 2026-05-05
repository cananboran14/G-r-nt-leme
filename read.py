import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Hata: Kamera açılamadı!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pencere başlığı artık boş
    cv2.imshow('', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() # Pencereyi düzgün kapatmak için bunu eklemen iyi olur
