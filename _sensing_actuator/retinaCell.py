''' creator: JJPark '''
''' editor : JYLee  '''

import cv2
import numpy as np

def negateMat(matrix, mul=0.5, pos=True):
    out = np.copy(matrix).reshape(-1)
    if pos:
        for i in range(matrix.size):
            if out[i] < 0:
                out[i] = out[i]*mul
    else:
        for i in range(matrix.size):
            if out[i] > 0:
                out[i] = -out[i]*mul
            else:
                out[i] = -out[i]
    return out.reshape(matrix.shape)


class cellLIF:
    def __init__(self, cellCount, thresFire=0.5, rateLeaky=0.9, rateAdapt=0.03, maxmAdapt=1.0):
        self.cellCount = cellCount
        self.cellAdapt = np.zeros(cellCount, dtype='float32')
        self.cellPoten = np.zeros(cellCount, dtype='float32')
        self.cellState = np.zeros(cellCount, dtype='int32')

        self.thresFire = thresFire
        
        self.rateLeaky = rateLeaky
        self.rateAdapt = rateAdapt
        self.thrsAdapt = maxmAdapt

    def update(self, signal):
        self.cellPoten = self.cellPoten + signal - self.cellAdapt * self.thrsAdapt
        self.cellAdapt = (self.cellAdapt * (1 - self.rateAdapt)) + (signal * self.rateAdapt)
        
        firedIdx = np.where(self.cellPoten > self.thresFire)
        self.cellPoten[firedIdx] = 0
        self.cellPoten *= self.rateLeaky
        
        self.cellState = np.zeros(self.cell_count, dtype='int32')
        self.cellState[firedIdx] = 1



class retinaCell:
    def __init__(self, shapeIn, shapeOut,\
                 thredFire=0.5, rateLeaky=0.9, rateAdapt=0.03, maxmAdapt=1):
        self.shapeIn  = shapeIn
        self.shapeOut = shapeOut
        
        self.cell = cellLIF(shapeOut, thredFire, rateLeaky, rateAdapt, maxmAdapt)

    def update(self, signal):
        return self.cell.cellState


class retinaOnOff(retinaCell):
    def __init__(self, shapeIn, shapeOut,\
                 thredFire=0.5, rateLeaky=0.9, rateAdapt=0.03, maxmAdapt=1):
        
        super().__init__(shapeIn, shapeOut, thredFire, rateLeaky, rateAdapt, maxmAdapt)

    def update(self, signal):
        frame = cv2.Laplacian(signal, cv2.CV_16S, ksize=1)
        frame = frame.astype('float32')/1024
        print(int(self.shapeOut[0]))
        signal[int(self.shapeOut[0]/2):] = negateMat(frame, 0.5, False)
        signal[:int(self.shapeOut[0]/2)] = negateMat(frame, 0.5, True)
        
        self.cell.update(signal)
        return self.cell.cellState
