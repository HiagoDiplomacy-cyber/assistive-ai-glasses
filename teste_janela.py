import cv2

while True:
    img = cv2.imread("C:/Windows/Web/Wallpaper/Windows/img0.jpg")

    if img is None:
        print("Imagem não carregou")
        break

    cv2.imshow("Teste", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
