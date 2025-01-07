import cv2

# Загрузите предобученный классификатор Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Инициализируйте видеозахват с веб-камеры (0 - это индекс камеры)
cap = cv2.VideoCapture(0)

while True:
    # Считывание кадра с видеопотока
    ret, frame = cap.read()

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Отрисовка прямоугольника вокруг каждого обнаруженного лица
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Отображение кадра с обнаруженными лицами
    cv2.imshow('Video', frame)

    # Завершение работы при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение видеозахвата и закрытие всех окон
cap.release()
cv2.destroyAllWindows()