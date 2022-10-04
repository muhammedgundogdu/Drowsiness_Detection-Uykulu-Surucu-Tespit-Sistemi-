from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import datetime as dt
import dlib
import cv2
import winsound
import matplotlib.pyplot as plt
import pyglet
import matplotlib.animation as animation
from EAR_calculator import *
import matplotlib.animation as animate #oran tablosu için gerekli kütüphane
from matplotlib import style 
import imutils 
import argparse 
from playsound import playsound
import os 
import csv
import pandas as pd
from datetime import datetime
import time



#fivethirtyeight türünde graph
style.use('fivethirtyeight')

#dataset oluşturma
def assure_path_exist(path):
    dir=os.path.dirname(path)
    if not os.path.dirname(dir):
        os.makedirs(dir)



#göz ve ağız oranları
ear_list=[]
total_ear=[]
mar_list=[]
total_mar=[]
ts=[]
total_ts=[]

#argüman ayrıştırma işlemleri
ap=argparse.ArgumentParser()
ap.add_argument("-p","--shape_predictor",required=True,  help="path to dlib's facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1, help="rasperi pi kamerası kullanıcan mı kullanmıcan mı")
args=vars(ap.parse_args())


#kulağın eşik değeri
EAR_THRESHOLD= 0.3
#göz açıp kapatana kadar geçen ardışık kare sayısı
CONSECUTIVE_FRAMES= 20
#bozulma değeri
MAR_THRESHOLD=27

BLINK_COUNT=0
FRAME_COUNT=0

#initialize dlib's face detector->detector && landmark predictor(yüzün nokta nokta tespiti)->predictor

print("[INFO]Loading the predictor....")
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(args["shape_predictor"])

#gözlerin indeksleri
(lstart, lend)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart,mend)=face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

#start video stream
print("[INFO]Loading camera...")
vs=VideoStream(usePiCamera=args["picamera"]>0).start()
time.sleep(2)

#uykulu ve esnemeli sayaçlar
assure_path_exist("/")
count_sleep=0
count_yawn=0

#loop ile bütün framelerden yüz tespiti
while True:
    #extract a frame
    frame=vs.read()
    cv2.putText(frame,"PRESS 'q' TO EXIT",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),3)
    #frame i boyutlandırma
    frame=imutils.resize(frame,width=500)
    #frameyi gri tonlamalı yapma
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #yüzü bulma
    rects=detector(frame,1)

    #detection bitti sıra geldi predictora(tahmin edici) 
    for (i,rect) in enumerate(rects):
        shape=predictor(gray,rect)
        #numpy arrayine dönüştürme
        shape=face_utils.shape_to_np(shape)

        #yüzü dikdörtgen içerisine alma
        (x,y,w,h)=face_utils.rect_to_bb(rect)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #sayı verme
        cv2.putText(frame,"Driver",(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        #göz ağız şekilleri
        leftEye=shape[lstart:lend]
        rightEye=shape[rstart:rend]
        mouth=shape[mstart:mend]
        #kulakların gözlere oranını hesaplama
        leftEAR=eye_aspect_ratio(leftEye)
        rightEAR=eye_aspect_ratio(rightEye)
        #iki kulağın ortalaması
        EAR=(leftEAR+rightEAR)/2.0
        #csv uzantılı data dosyası
        ear_list.append(EAR)

        ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
        #Dış bükeyi görselleştirme
        leftEyeHull= cv2.convexHull(leftEye)
        rightEyeHull=cv2.convexHull(rightEye)
        #Konturler
        cv2.drawContours(frame, [leftEyeHull],-1,(0,255,0),1)
        cv2.drawContours(frame, [rightEyeHull],-1,(0,255,0),1)
        cv2.drawContours(frame, [mouth],-1,(0,255,0),1)

        MAR=mouth_aspect_ratio(mouth)
        mar_list.append(MAR/10)

        #if EAR<EAR_THRESHOLD göz kırpma gerçekleşti
        #gözün kapalı kaldığı kareler sayılır

        if EAR<EAR_THRESHOLD:
            FRAME_COUNT +=1

            cv2.drawContours(frame,[leftEyeHull],-1,(0,0,255),1)    
            cv2.drawContours(frame,[rightEyeHull],-1,(0,0,255),1)    
            
            if FRAME_COUNT>= CONSECUTIVE_FRAMES:
                count_sleep+=1
                playsound('sound files/alarm.mp3')
                cv2.putText(frame,"UYKULU SURUCU UYARISI",(270,30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
                
        else:
            if FRAME_COUNT>=CONSECUTIVE_FRAMES:
                playsound('sound files/warning.mp3')
            FRAME_COUNT=0
            
        
        #esneme durumu kontrolü
        if MAR > MAR_THRESHOLD:
            count_yawn+=0.5
            cv2.drawContours(frame,[mouth],-1,(0,0,255),1)
            cv2.putText(frame,"UYKULU SURUCU UYARISI", (270,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
            #frami uykulu sürücü olduğuna dair datasete ekleme
            #cv2.imwrite("dataset/frame_yawn%d.jpg"%count_yawn,frame)
            playsound('sound files/alarm.mp3')
            
    #frami show etme
    cv2.imshow("Result",frame)
    key=cv2.waitKey(1) & 0xFF

    if key== ord('q'):
        break
cv2.destroyAllWindows()
vs.stop()











