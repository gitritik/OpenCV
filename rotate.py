#Output---->Rotating version of video 'rotating.avi'

import array
import cv2 as cv
import numpy as np



videoFile = "Video/street.mp4"
vidcap = cv.VideoCapture(videoFile)
success, image = vidcap.read()


fps = vidcap.get(cv.CAP_PROP_FPS) 



imlst=[]
while success:
    frameId = int(round(vidcap.get(1))) 
    success, image = vidcap.read()
    imlst.append(image)

height,width,layers=imlst[0].shape
size=(width,height)

def rotate (img,angle,rotPoint=None):

    h,w=img.shape[:2]

    if rotPoint==None:
        rotPoint=(w//2,h//2)

    rotMat=cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimentions=(w,h)

    return cv.warpAffine(img,rotMat,dimentions)


rot_lst=[]
ang=0
for q in imlst:
    if type(q)==np.ndarray and type(q)!=None :


        rotated=rotate(q,ang)
        rot_lst.append(rotated)
        ang=ang+2
print(size)
out1=cv.VideoWriter("Outputs/rotating.avi",cv.VideoWriter_fourcc(*'DIVX'),fps,size)
for j in rot_lst:
    out1.write(j)
out1.release()
vidcap.release()
print("Complete")