import numpy as np
import random

class columnalLayer:
    def __init__(self, cellType = 'NR', cellCount = 1, proxInputSize = 2):
        # Membrane Characteristics
        self.cellType  = cellType
        self.cellCount = cellCount
        
        self.axonState = 0 # [rest, depo, fire, brst]
        
        # Dendrite Development
        self.proxInSz = proxInputSize
        self.proxDend = np.random.rand(proxInputSize) 


    def update(self, proxInput, thresFire, thresSynProx,\
               longEffic, shortEffic, homeoEffic, learnFlag = False,\
               verbose = False):

        # Initialize Axon State
        self.axonState[:] = 0
        
        # Integrate to Fire
        proxDend_ = np.where( (self.proxDend >= thresSynProx), self.proxDend, 0 )
        proxScore = proxDend_ @ proxInput.reshape(-1)
        proxFireIndex = np.where( proxScore >= thresFire )[0]
        np.random.shuffle(proxFireIndex)
        proxFireIndex = proxFireIndex[:min(1, len(proxFireIndex))]
        self.axonState[proxFireIndex] = 1
            
        # Synaptic Efficacy
        if(learnFlag):
            ## Proximal Input
            for src in proxFireIndex:
                for dst in np.where( proxInput.reshape(-1) >  0 ): # f-f
                    self.proxDend[src][dst] *= (1+longEffic)
                for dst in np.where( proxInput.reshape(-1) == 0 ): # r-f
                    self.proxDend[src][dst] *= (1-shortEffic)

            for src in list(set(range(self.cellCount)) - set(proxFireIndex)):
                for dst in np.where( proxInput.reshape(-1) >  0 ): # f-r
                    self.proxDend[src][dst] *= (1-shortEffic)
                for dst in np.where( proxInput.reshape(-1) == 0 ): # r-r
                    self.proxDend[src][dst] *= (1+0)

            self.proxDend = np.clip(self.proxDend, 0.2, 1)
            
    def save(self, dataPath='model/_tmp/tmp.npz'):
        np.savez(dataPath,\
                 cellType = self.cellType, cellCount = self.cellCount,\
                 axonState = self.axonState,\
                 proxInSz = self.proxInSz, proxDend = self.proxDend)

    def load(self, dataPath='model/_tmp/tmp.npz'):
        npzData = np.load(dataPath)
        
        self.cellType  = npzData['cellType']
        self.cellCount = npzData['cellCount']
        self.axonState = npzData['axonState']

        self.proxInSz = npzData['proxInSz']
        self.proxDend = npzData['proxDend']


class simpleLayer(columnalLayer): 
    def __init__(self, cellType = 'PC', cellCount = 50*50,\
                 fireRate = 0.02, depoRate = 0.05,\
                 proxInputSize=30*30, distInputSize=50*50):
        
        # Membrane Characteristics
        self.cellType  = cellType
        self.cellCount = cellCount

        self.fireCount = int(cellCount * fireRate)
        self.depoCount = int(cellCount * depoRate)
        
        self.restMemo = np.zeros(cellCount)
        
        # [rest, depo, fire, brst]
        initFire = random.sample(range(0, cellCount), self.fireCount)
        initDepo = random.sample(range(0, cellCount), self.depoCount)
        self.axonState = np.zeros(cellCount, dtype='int32')
        self.axonState[initFire] = 2 
        self.axonState[initDepo] = 1
        
        # Dendrite Development
        self.proxInSz = proxInputSize
        self.distInSz = distInputSize
        
        self.proxDend = np.random.rand(cellCount, proxInputSize) 
        self.distDend = np.random.rand(cellCount, distInputSize) 

    def update(self, proxInput, distInput,\
               thresFire, thresSynProx, thresDepo, thresSynDist,\
               longEffic, shortEffic, homeoEffic, learnFlag = False,\
               verbose = False):
        
        # Integrate to Fire
        self.axonState[:] = 0

        ## Distal Input to Depolarize
        distDend_ = np.where( (self.distDend >= thresSynDist), self.distDend, 0 )
        distScore = distDend_ @ distInput.reshape(-1)
        distDepoIndex = np.where( distScore >= thresDepo )[0]
        np.random.shuffle(distDepoIndex)
        distDepoIndex = distDepoIndex[:min(self.depoCount, len(distDepoIndex))]
        self.axonState[distDepoIndex] = 1

        ## Proximal Input To Fire (If Predicted, then Burst)
        proxDend_ = np.where( (self.proxDend >= thresSynProx), self.proxDend, 0 )
        proxScore = proxDend_ @ proxInput.reshape(-1)
        proxDepoIndex = np.where( proxScore >= thresFire )[0]
        np.random.shuffle(proxDepoIndex)
        proxFireIndex = [ value for value in proxDepoIndex \
                          if value in np.where(self.axonState == 1)[0] ][:self.fireCount]
        proxDepoIndex = [ value for value in proxDepoIndex \
                          if value in np.where(self.axonState == 0)[0] ][:max(0, self.fireCount-len(proxFireIndex))]
        self.axonState[proxDepoIndex] = 2
        self.axonState[proxFireIndex] = 3

        # Resource Usage (Homeostasis)
        self.restMemo[ np.where(self.axonState == 0) ] += 1
        self.restMemo[ np.where(self.axonState == 1) ] -= 1
        self.restMemo[ np.where(self.axonState >= 2) ]  = 0
        
        self.restMemo[ np.where(self.restMemo  >= 5) ]  = 5
        self.restMemo[ np.where(self.restMemo  <= 0) ]  = 0
            
        # Synaptic Efficacy
        if(learnFlag):
            ## Proximal Input
            for src in proxFireIndex + proxDepoIndex:
                for dst in np.where( proxInput.reshape(-1) >  0 ): # f-f
                    self.proxDend[src][dst] *= (1+longEffic)
                for dst in np.where( proxInput.reshape(-1) == 0 ): # r-f
                    self.proxDend[src][dst] *= (1-shortEffic)

            for src in list(set(range(self.cellCount)) - set(proxFireIndex) - set(proxDepoIndex)):
                for dst in np.where( proxInput.reshape(-1) >  0 ): # r-f
                    self.proxDend[src][dst] *= (1-shortEffic) 
                for dst in np.where( proxInput.reshape(-1) == 0 ): # r-r
                    self.proxDend[src][dst] *= (1+0)

            ## Distal Input
            ### Right Prediction 
            for src in proxFireIndex: # ?-d-f
                for dst in np.where( distInput.reshape(-1) >  0 ): # f-d-f
                    self.distDend[src][dst] *= (1+longEffic)
                for dst in np.where( distInput.reshape(-1) == 0 ): # r-d-f
                    self.distDend[src][dst] *= (1+homeoEffic)

            ### Wrong Prediction 
            for src in list(set(distDepoIndex) - set(proxFireIndex)):  # ?-d-r
                for dst in np.where( distInput.reshape(-1) >  0 ): # f-d-r
                    self.distDend[src][dst] *= (1-longEffic)
                for dst in np.where( distInput.reshape(-1) == 0 ): # r-d-r
                    self.distDend[src][dst] *= (1+shortEffic)
                    
            ### No Prediction
            for src in list(set(range(self.cellCount)) - set(distDepoIndex)): 
                for dst in np.where( distInput.reshape(-1) >  0 ): # f-r-r
                    self.distDend[src][dst] *= (1-shortEffic)
                for dst in np.where( distInput.reshape(-1) == 0 ): # r-r-r
                    self.distDend[src][dst] *= (1+0)
            
            self.proxDend = np.clip(self.proxDend, 0.2, 1)
            self.distDend = np.clip(self.distDend, 0.2, 1)
            
    def save(self, dataPath='model/_tmp/tmp.npz'):
        np.savez(dataPath,\
                 cellType = self.cellType, cellCount = self.cellCount,\
                 axonState = self.axonState,\
                 fireCount = self.fireCount, depoCount = self.depoCount,\
                 proxDend = self.proxDend, distDend = self.distDend,\
                 proxInSz = self.proxInSz, distInSz = self.distInSz)

    def load(self, dataPath='model/_tmp/tmp.npz'):
        npzData = np.load(dataPath)
        
        # Membrane Characteristics
        self.cellType  = npzData['cellType']
        self.cellCount = npzData['cellCount']
        self.axonState = npzData['axonState']
        
        self.fireCount = npzData['fireCount']
        self.depoCount = npzData['depoCount']

        # Dendrite Development        
        self.proxInSz = npzData['proxInSz']
        self.distInSz = npzData['distInSz']
        
        self.proxDend = npzData['proxDend']
        self.distDend = npzData['distDend']


class pyramidalLayer(columnalLayer): # Excitatory Neuron
    def __init__(self, cellType = 'PC', cellCount = 50*50,\
                 fireRate = 0.02, depoRate = 0.05,\
                 proxInputSize=20*20, distInputSize=50*50, inhiInputSize=50*50, apicInputSize=50*50):
        
        # Membrane Characteristics
        self.cellType  = cellType
        self.cellCount = cellCount

        self.fireCount = int(cellCount * fireRate)
        self.depoCount = int(cellCount * depoRate)
        
        initFire = random.sample(range(0, cellCount), self.fireCount)
        self.axonState = np.zeros(cellCount, dtype='int32')
        self.axonState[initFire] = 1 
        
        # Dendrite Development
        self.proxInSz = proxInputSize
        self.distInSz = distInputSize
        self.inhiInSz = inhiInputSize
        self.apicInSz = apicInputSize
        
        self.proxDend = np.random.rand(cellCount, proxInputSize) 
        self.distDend = np.random.rand(cellCount, distInputSize) 
        self.inhiDend = np.random.rand(cellCount, inhiInputSize) 
        self.apicDend = np.random.rand(cellCount, apicInputSize) 

    def update(self, proxInput, distInput, inhiInput, apicInput,\
               thresDepo, thresInhi, thresSyn,\
               longEffic, shortEffic, homeoEffic, learnFlag = False,\
               verbose = False):

        # Initiate Axon State
        self.axonState[:] = 0
        
        # Integrate to Fire
        ## Distal Input to Depolarize
        distDend_ = np.where( (self.distDend >= thresSyn), self.distDend, 0 )
        distScore = distDend_ @ distInput.reshape(-1)
        distDepoIndex = np.where( distScore >= thresDepo )[0]
        np.random.shuffle(distDepoIndex)
        distDepoIndex = distDepoIndex[:min(self.depoCount, len(distDepoIndex))]

        ## Proximal Input to Depolarize
        proxDend_ = np.where( (self.proxDend >= thresSyn), self.proxDend, 0 )
        proxScore = proxDend_ @ proxInput.reshape(-1)
        proxDepoIndex = np.where( proxScore >= thresDepo )[0]
        np.random.shuffle(proxDepoIndex)
        proxDepoIndex = proxDepoIndex[:min(self.depoCount, len(proxDepoIndex))]

        ## Inhibition Input to Polarize
        inhiDend_ = np.where( (self.proxDend >= thresSyn), self.proxDend, 0 )
        inhiScore = inhiDend_ @ inhiInput.reshape(-1)
        inhiPolaIndex = np.where( inhiScore >= thresInhi )[0]
        np.random.shuffle(inhiPolaIndex)
        inhiPolaIndex = proxDepoIndex[:min(self.depoCount, len(inhiPolaIndex))]

        ## Fire
        proxFireIndex = [ value for value in proxDepoIndex \
                          if value in np.where(self.axonState == 1)[0] ][:self.fireCount]
        self.axonState[proxFireIndex] = 1
        

        # Synaptic Efficacy
        if(learnFlag):
            ## Proximal Input
            for src in proxFireIndex + proxDepoIndex:
                for dst in np.where( proxInput.reshape(-1) >  0 ): # f-f
                    self.proxDend[src][dst] *= (1+longEffic)
                for dst in np.where( proxInput.reshape(-1) == 0 ): # r-f
                    self.proxDend[src][dst] *= (1-shortEffic)

            for src in list(set(range(self.cellCount)) - set(proxFireIndex) - set(proxDepoIndex)):
                for dst in np.where( proxInput.reshape(-1) >  0 ): # r-f
                    self.proxDend[src][dst] *= (1-shortEffic) 
                for dst in np.where( proxInput.reshape(-1) == 0 ): # r-r
                    self.proxDend[src][dst] *= (1+0)

            ## Distal Input
            ### Right Prediction 
            for src in proxFireIndex: # ?-d-f
                for dst in np.where( distInput.reshape(-1) >  0 ): # f-d-f
                    self.distDend[src][dst] *= (1+longEffic)
                for dst in np.where( distInput.reshape(-1) == 0 ): # r-d-f
                    self.distDend[src][dst] *= (1+homeoEffic)

            ### Wrong Prediction 
            for src in list(set(distDepoIndex) - set(proxFireIndex)):  # ?-d-r
                for dst in np.where( distInput.reshape(-1) >  0 ): # f-d-r
                    self.distDend[src][dst] *= (1-longEffic)
                for dst in np.where( distInput.reshape(-1) == 0 ): # r-d-r
                    self.distDend[src][dst] *= (1+shortEffic)
                    
            ### No Prediction
            for src in list(set(range(self.cellCount)) - set(distDepoIndex)): 
                for dst in np.where( distInput.reshape(-1) >  0 ): # f-r-r
                    self.distDend[src][dst] *= (1-shortEffic)
                for dst in np.where( distInput.reshape(-1) == 0 ): # r-r-r
                    self.distDend[src][dst] *= (1+0)

            ## Interneurons
            
            self.proxDend = np.clip(self.proxDend, 0.2, 1)
            self.distDend = np.clip(self.distDend, 0.2, 1)
            self.inhiDend = np.clip(self.inhiDend, 0.2, 1)
            self.inhiDend = np.clip(self.inhiDend, 0.2, 1)
            

class stellateLayer(columnalLayer): # Interneurons
    def __init__(self, cellType = 'SS', cellCount = 50*50,\
                 fireRate = 0.02, depoRate = 0.05,\
                 proxInputSize=30*30, distInputSize=50*50, apicInputSize=1):
        
        # Membrane Characteristics
        self.cellType  = cellType
        self.cellCount = cellCount

        self.fireCount = int(cellCount * fireRate)
        self.depoCount = int(cellCount * depoRate)
        
        self.restMemo = np.zeros(cellCount)
        
        # [rest, depo, fire, brst]
        initFire = random.sample(range(0, cellCount), self.fireCount)
        initDepo = random.sample(range(0, cellCount), self.depoCount)
        self.axonState = np.zeros(cellCount, dtype='int32')
        self.axonState[initFire] = 2 
        self.axonState[initDepo] = 1
        
        # Dendrite Development
        self.proxInSz = proxInputSize
        self.distInSz = distInputSize
        
        self.proxDend = np.random.rand(cellCount, proxInputSize) 
        self.distDend = np.random.rand(cellCount, distInputSize) 


    def update(self, proxInput, distInput, apicInput):
        pass

