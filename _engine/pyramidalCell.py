''' design : JYLee '''

class cellBody:
    def __init__(self):
        self.cellType
        self.cellStat #(rest, depo, fire, brst)

        self.thresDepo
        self.thresFire

class dendriticArbor:
    def __init__(self):
        self.dendStat #(rest, fire)

        self.src ## source Cell Index
        self.dst ## target Cell Index
        self.syn ## number of Synapses


class pyramidalCell:
    def __init__(self, bodyParam, targetArray):
        # Membrane Characteristics
        self.cellBody = cellBody(bodyParam)
        
        # Wire Dendrites
        self.proxDendrite = [ dendriticArbor() for proxCount ]
        self.distDendrite = [ dendriticArbor() for distCount ]
        self.apicDendrite = [ dendriticArbor() for apicCount ]

        for target in targetArray:
            for dendrite in dendriteArray:
                dendrite.src = self
                dendrite.dst = target ## Topology
                dendrite.syn = random((1,10))

    def update(self, signal):
        # State Initiation
        if( self.cellBody.cellStat == fire)   state = rest
        elif( self.cellBody.cellStat == brst) state = depo

        # Integrate Signal
        dendriteSig = dendriteArray[ where(target.syn > 7) ]
        dendriteSig = dendriteSig[ : dendrite.rfSize ] ## Sparseness
        dendriteScr = sum(dendriteSig.target.state)

        # State Update
        if( apicScore > thresDepo )
            state = depo

        if( distScore > thresFire && state == depo)
            state = fire
        elif( distScore > thresDepo && state == rest)
            state = depo

        if( proxScore > thresFire && state == depo)
            state = brst
        elif( proxScore > thresFire)
            state = fire

        # Synaptic Efficacy
        for dendrite in dendriteArray:
            if( dendrite.target.state >= fire)
                dendrite.syn += 1
            else
                dendrite.syn -= 1

        # Homeostasis
        for dendrite in dendriteArray:
            if( dendrite.syn < 1)
                dendrite.syn = 5
        
    
