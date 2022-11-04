#Output-First scans the face from webcam for sample collection.Then develos pretrained detection model from the sample and then detects the availability of thst face in camera from pre trained module from scanned images.
import os
import array
import cv2 as cv
import numpy as np



haar_cas=cv.CascadeClassifier('haar_f.xml')


imlst=[]



print('show your face with multiple angles on camera for sample scanning')  

vidcap = cv.VideoCapture(0)
success, image = vidcap.read()
fps = vidcap.get(cv.CAP_PROP_FPS)


imlst=[]
tc_count=0

while success and tc_count<=200:
    frameId = int(round(vidcap.get(1)))
    success, image = vidcap.read()
    image=cv.flip(image,1)
    tc_count+=1
    imlst.append(image)


print("thank you")

train_sample=[]
labels=[]
label=0
for n in range(len(imlst)):
    if type(imlst[n])==np.ndarray and type(imlst[n])!=None :
        gray=cv.cvtColor(imlst[n],cv.COLOR_BGR2GRAY)


        face_cor=haar_cas.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)
        for (x,y,w,h) in face_cor:

            roi=gray[y:y+h,x:x+w]
            train_sample.append(roi)
            labels.append(label)
            label+=1

train_sample=np.array(train_sample,dtype='object')
labels=np.array(labels)



face_recogniser=cv.face.LBPHFaceRecognizer_create()

face_recogniser.train(train_sample,labels)

vidcap = cv.VideoCapture(0)
success, image = vidcap.read()
gray_test=[]
while success:
    frameId = int(round(vidcap.get(1)))
    success, image = vidcap.read()
    image=cv.flip(image,1)
    if type(image)==np.ndarray and type(image)!=None :
        gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        face_cor=haar_cas.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        for (x,y,w,h) in face_cor:
            roi=gray[y:y+h,x:x+w]
            gray_test.append(roi)
        finding=image   
        for face in gray_test:
            label,confidence=face_recogniser.predict(face)
            if confidence>80:
                cv.rectangle(finding,(x,y),(x+w,y+h),(0,255,0),thickness=2)
                cv.putText(finding,"Person",(x,int(y+(25*h/200))),cv.FONT_HERSHEY_SIMPLEX,fontScale=(1*w/200),color=(0,250,0),thickness=1,lineType=cv.LINE_AA)
                print("Confidence is",confidence)
        cv.imshow("recog_2_out", finding)
        cv.waitKey(1)
                
        gray_test=[]