#detects face on camera
import array
import cv2 as cv
import math
import numpy as np
import copy
haar_cas=cv.CascadeClassifier('haar_f.xml')




####
vidcap = cv.VideoCapture(0)
success, image = vidcap.read()
fps = vidcap.get(cv.CAP_PROP_FPS)
######



change=int(fps)



def face_detect(img):
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_cor=haar_cas.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    for(x,y,w,h) in face_cor:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,120,250),thickness=2)
        cv.putText(img,"fps=%d"%fps,(30,30),cv.FONT_HERSHEY_PLAIN,fontScale=1,color=(0,0,0),thickness=1,lineType=cv.LINE_AA)
        cv.putText(img,"Person",(x,int(y+(25*h/200))),cv.FONT_HERSHEY_SIMPLEX,fontScale=(1*w/200),color=(0,250,0),thickness=1,lineType=cv.LINE_AA)
        print(w)
    
    return img
    
    


while success:
    frameId = int(round(vidcap.get(1)))
    success, image = vidcap.read()
    image=cv.flip(image,1)
    cv.imshow("cam",face_detect(image))
    cv.waitKey(1)