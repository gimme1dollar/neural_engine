import numpy as np
import cv2

class visualizeRF:
    def __init__(self, cellArray=np.zeros((50*50))):
        self.cellArray = cellArray
        
    def save(self, dataPath = "data/", dataSize=(28,28)):
        w , h  = int(np.ceil(np.sqrt(cell.cell_count))), int(np.ceil(cell.cell_count / w))
        ww, hh = dataSize
        
        rf_img  = np.full( ( h*(hh+5), w*(ww+5) ), 80, dtype='uint8' )
        default = np.full( ( h*(hh+5), w*(hh+5) ), 80, dtype='uint8' )
        
        for ii in range(cell.cell_count):
            rf = np.zeros(hh*ww)
            proxRFIndex = np.where(self.cellArray.proxDend[ii] >= 0.7)
            rf[self.cellArray.proxDend[ii][proxRFIndex]] = 255
            rf = rf.reshape((hh*2, ww))

            top  = int(ii / w) * (hh+5)
            left = int(ii % w) * (ww+5)
            rf_img[ top : top+hh , left : left+ww ] = rf[:hh]
            
            resImage = cv2.merge([default, rf_img])
            
        cv2.imwrite(f"{dataPath}.bmp", resImage)
