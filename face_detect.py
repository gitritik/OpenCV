#output--->gives the video with face detection frame by frame

import array
import cv2 as cv
import numpy as np


videoFile = "Video/street.mp4"
vidcap = cv.VideoCapture(videoFile)
success, image = vidcap.read()


fps = vidcap.get(cv.CAP_PROP_FPS) # Gets the frames per second

imlst=[]
while success:
    frameId = int(round(vidcap.get(1))) 
    success, image = vidcap.read()
    imlst.append(image)

height,width,layers=imlst[0].shape
size=(width,height)


haar_cas=cv.CascadeClassifier('haar_f.xml')
det_lst=[]
b=0
for q in imlst:
    if type(q)==np.ndarray and type(q)!=None :
        gray=cv.cvtColor(q,cv.COLOR_BGR2GRAY)
        face_cor=haar_cas.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

        for(x,y,w,h) in face_cor:
            cv.rectangle(q,(x,y),(x+w,y+h),(0,255,0),thickness=2)
            det_lst.append(q)
        print(b)#printing current frame going through face detection
        b=b+1

print(size)
out1=cv.VideoWriter("Outputs/faces.avi",cv.VideoWriter_fourcc(*'DIVX'),fps*3,size)
for j in det_lst:
    out1.write(j)
out1.release()
vidcap.release()
print("Complete")