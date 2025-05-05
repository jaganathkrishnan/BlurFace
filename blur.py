import cv2
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in face:
        image = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        image[y:y+h, x:x+w] = cv2.GaussianBlur(image[y:y+h, x:x+w], (99, 99), 30)
    cv2.imshow('Face Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()   
