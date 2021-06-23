import os
import numpy as np
import csv
import cv2
from time import gmtime, strftime

#from _controller import retinaCell
from _engine import columnalLayer

L4cellNum = 5
L4 = columnalLayer.simpleLayer(cellCount = L4cellNum*L4cellNum,\
                                 fireRate = 0.1,\
                                 depoRate = 0.1,\
                                 proxInputSize = L4cellNum*L4cellNum,\
                                 distInputSize = L4cellNum*L4cellNum)
#L4.load('model/2_ballStable.npz')
brstRastor = np.zeros((L4.cellCount, 50))

ballXX, ballYY = int(L4cellNum/2), int(L4cellNum/2)

while(True):
    # Data
    ball = np.zeros((L4cellNum,L4cellNum), dtype='uint8')
    ball[ballXX][ballYY] = 255
    
    # L4
    cellSt = np.copy(L4.axonState)
    L4.update(proxInput = ball/255, thresFire = 0.5, \
              distInput = cellSt, thresDepo = 1,\
              thresSynProx = 0.7, thresSynDist = 0.7,\
              learnFlag = True,\
              longEffic = 0.3, shortEffic = 0.1, homeoEffic = 0.00,\
              
              verbose = False)


    # Plot
    output = L4.axonState/3
    output = np.asarray(output.reshape(L4cellNum,L4cellNum)*255, dtype=np.uint8)

    proxSynapse = np.zeros((L4cellNum**2,L4cellNum**2))
    proxSynapse[np.where(L4.proxDend >= 0.7)] = 1
    #proxSynapse = L4.proxDend
    proxSynapse = np.asarray(proxSynapse.reshape(L4cellNum**2,L4cellNum**2)*255, dtype=np.uint8)

    distSynapse = np.zeros((L4cellNum**2,L4cellNum**2))
    distSynapse[np.where(L4.distDend >= 0.7)] = 1
    #distSynapse = L4.distDend
    distSynapse = np.asarray(distSynapse.reshape(5**2,5**2)*255, dtype=np.uint8)

    for column in range(50-1):
        brstRastor[:, 50-2-column] = brstRastor[:, 50-3-column]
    brstRastor[np.where(L4.axonState.reshape(-1)>=3), 0] = 1

    
    cv2.namedWindow('raw', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('raw', (300,300))
    cv2.imshow('raw', ball)
    
    cv2.namedWindow('out', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('out', (300,300))
    cv2.imshow('out', output)
    
    cv2.namedWindow('prox', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('prox', (300,300))
    cv2.imshow('prox', proxSynapse)
    
    cv2.namedWindow('dist', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dist', (300,300))
    cv2.imshow('dist', distSynapse)

    cv2.namedWindow('rast', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('rast', (300,300))
    cv2.imshow('rast', brstRastor)
    
    
    # Controll
    k = cv2.waitKey(0) & 0xFF
    if k == ord('8'):
        ballXX = (ballXX-1)%5
    if k == ord('2'):
        ballXX = (ballXX+1)%5
    if k == ord('4'):
        ballYY = (ballYY-1)%5
    if k == ord('6'):
        ballYY = (ballYY+1)%5

    if k == ord('5'):
        pass
    if k == ord('b'):
        L4.save('model/dummy.npz')
        break
