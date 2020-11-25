import cv2

cap = cv2.VideoCapture('video.avi')
car_cascade = cv2.CascadeClassifier('cars.xml')

while True:
    ret,frames = cap.read()
    gray = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray,1.1,9)
    for(x,y,w,h) in cars:
        cv2.rectangle(frames,(x,y),(x+h,y+w),(10,50,0),2)
        cv2.rectangle(frames,(x,y-40),(x+w ,y),(10,50,0),-2)
        cv2.putText(frames,'Car Detected',(x,y-10),cv2.FONT_ITALIC,0.7,(255,255,255),2)

    frames = cv2.resize(frames,(600,400))
    cv2.imshow('CAR DETECTION SYSTEM', frames)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()