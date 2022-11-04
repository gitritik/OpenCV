
import array
import cv2 as cv
import math
import numpy as np
import copy

####
vidcap = cv.VideoCapture(0)
success, image = vidcap.read()
fps = vidcap.get(cv.CAP_PROP_FPS)
######



change=int(fps)


def canny_2(img,status,count):
        blank=np.zeros(img.shape[:2],dtype="uint8")
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        canny=cv.Canny(gray,10,30)
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
        
        return (final,status,count)

status="g"
count=0
while success:
    frameId = int(round(vidcap.get(1)))
    success, image = vidcap.read()
    image=cv.flip(image,1)
    im,status,count=canny_2(image,status,count)
    cv.imshow("cam",im)
    cv.waitKey(1)
    