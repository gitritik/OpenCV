#Output-Detects the availability of face in each nth frame of the video from pre trained module from given sample images.Output in recog_2_out folder
import os
import array
import cv2 as cv
import numpy as np



haar_cas=cv.CascadeClassifier('haar_f.xml')


imlst=[]




name=os.listdir(r"sample")[0]

path1=os.path.join(r"sample",name)

for img in os.listdir(path1):
    img_path=os.path.join(path1,img)
    img_array=cv.imread(img_path)
    imlst.append(img_array)

# print(imlst)
# gray_lst=[]

train_sample=[]
labels=[]
label=0
for n in range(len(imlst)):
    if type(imlst[n])==np.ndarray and type(imlst[n])!=None :
        gray=cv.cvtColor(imlst[n],cv.COLOR_BGR2GRAY)


        face_cor=haar_cas.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)
        for (x,y,w,h) in face_cor:

            roi=gray[y:y+h,x:x+w]
            cv.imwrite("FolderSeconds/frame%la.jpg" % int(label), roi)
            train_sample.append(roi)
            labels.append(label)
            label+=1

train_sample=np.array(train_sample,dtype='object')
labels=np.array(labels)



face_recogniser=cv.face.LBPHFaceRecognizer_create()

face_recogniser.train(train_sample,labels)


################################
testFile = "Video/test_video.mp4"
vidcap = cv.VideoCapture(testFile)
success, image = vidcap.read()


fps = vidcap.get(cv.CAP_PROP_FPS) 


test_lst=[]
while success:
    frameId = int(round(vidcap.get(1)))
    success, image = vidcap.read()
    test_lst.append(image)
    
gray_test=[]
for q in range(len(test_lst)):
    if type(test_lst[q])==np.ndarray and type(test_lst[q])!=None :
        gray=cv.cvtColor(test_lst[q],cv.COLOR_BGR2GRAY)
        face_cor=haar_cas.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=8)
        for (x,y,w,h) in face_cor:

            roi=gray[y:y+h,x:x+w]
            gray_test.append(roi)
        for face in gray_test:
            label,confidence=face_recogniser.predict(face)
            if confidence>100:
                print(confidence,q//fps)
                finding=test_lst[q]
                cv.rectangle(finding,(x,y),(x+w,y+h),(0,255,0),thickness=2)
                cv.imwrite("recog_2_out/frame%la.jpg" % int(q), finding)
                q+=int(fps)


        gray_test=[]

