from ultralytics import YOLO
import cv2
import pyttsx3
import threading
import time

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

voz = pyttsx3.init()

objetos_falados = {}

def falar(texto):
    voz.say(texto)
    voz.runAndWait()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    nomes_detectados = []

    for result in results:
        if result.boxes:
            for box in result.boxes:
                classe = int(box.cls[0])
                nome = model.names[classe]
                nomes_detectados.append(nome)

    tempo_atual = time.time()

    for nome in nomes_detectados:
        if nome not in objetos_falados or tempo_atual - objetos_falados[nome] > 3:
            threading.Thread(target=falar, args=(nome,)).start()
            objetos_falados[nome] = tempo_atual

    annotated_frame = results[0].plot()

    cv2.imshow("Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
