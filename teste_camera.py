import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERRO na Cãmera")
        break

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.DestroyAllWindows()