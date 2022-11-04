
#Output ---->Black and white version of video"gray.avi",video with contour edges"canny.avi"
import array
import cv2 as cv
import math
import numpy as np
import copy




videoFile = "Video/street.mp4"#sample file
vidcap = cv.VideoCapture(videoFile)
success, image = vidcap.read()


fps = vidcap.get(cv.CAP_PROP_FPS)



imlst=[] #list with each individual frame
while success:
    frameId = int(round(vidcap.get(1)))
    success, image = vidcap.read()
    imlst.append(image)



height,width,layers=imlst[0].shape
size=(width,height)
can_lst=[]
gray_lst=[]
for q in imlst:
    if type(q)==np.ndarray and type(q)!=None :
        gray=cv.cvtColor(q,cv.COLOR_BGR2GRAY)
        canny=cv.Canny(gray,50,175)
        can_f=cv.cvtColor(canny,cv.COLOR_GRAY2BGR)
        gray_f=cv.cvtColor(gray,cv.COLOR_GRAY2BGR)
        cv.imshow('a',can_f)
        cv.waitKey(2)
        can_lst.append(can_f)
        gray_lst.append(gray_f)
print(size)
out1=cv.VideoWriter("canny.avi",cv.VideoWriter_fourcc(*'DIVX'),fps,size)
for j in can_lst:#compiling the output from modified frames in list
    out1.write(j)
out1.release()

out2=cv.VideoWriter("Outputs/gray.avi",cv.VideoWriter_fourcc(*'DIVX'),fps,size)
for k in gray_lst:
    out2.write(k)

vidcap.release()
out2.release()
print("Complete")
