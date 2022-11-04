#Output--->Disco version of videos'contour edges 'canny_2.avi'

import array
import cv2 as cv
import numpy as np





videoFile = "Video/dance.mp4"
vidcap = cv.VideoCapture(videoFile)
success, image = vidcap.read()


fps = vidcap.get(cv.CAP_PROP_FPS)



height,width,layers=vidcap.read()[1].shape
size=(width,height)
count=0
change=int(fps)
status="b"
out1=cv.VideoWriter("Outputs/canny_2.avi",cv.VideoWriter_fourcc(*'DIVX'),fps,size)
imlst=[]
blank=np.zeros((height,width),dtype="uint8")    
while success:
    success, image = vidcap.read()
    imlst.append(image)
    if type(image)==np.ndarray and type(image)!=None :
        gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        canny=cv.Canny(gray,20,50)
        can_f=cv.cvtColor(canny,cv.COLOR_GRAY2BGR)
        b,g,r=cv.split(can_f)
        if status=="b":
            final=cv.merge([b,blank,blank])
        elif status=="g":
            final=cv.merge([blank,g,blank])
        elif status=="r":
            final=cv.merge([blank,blank,r])
        count+=1
        if count%change<=int(fps//3):
            status="g"
        elif count%change>=2*int(fps//3):
            status="b"
        else:
            status="r"
        out1.write(final)
        print(count)    



print(size)
out1.release()

vidcap.release()
print("Complete")