import cv2
import pyttsx3

# iniciar voz
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# iniciar câmera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    if not ret:
        break

    cv2.imshow("Assistive AI Glasses", frame)

    # exemplo simples
    speak("Camera active")

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
