# ANEM
Online Self-Rewarded Hierarchical Feature Extracting Algorithm

## Algorithm (Concept)
### System
- Controller : Data Encoder, etc.
- Engine : Pyramidal-cell-inspired Online Self-rewarded Feature Extractor
- Visualizer : Receptive-field Visulaization, etc.

### Engine
##### Pyramidal Cell
- Cell Body : Membrane Characteristics
- Dendrites : Connection & Synaptic Plasticity

##### Cortical Column
- 5-Layers : Hierarchical Feature (Vector) Extraction
- Attention : Reward-based Reduction Reconstruction

## Result

### Spatial Feature
#### [Receptive Field Construction]
(initial condition)
![rf1](https://user-images.githubusercontent.com/20160685/88382430-62f95880-cde3-11ea-8588-303f362767d3.png)  
(trained RFs)
![rf3](https://user-images.githubusercontent.com/20160685/88382473-7efcfa00-cde3-11ea-93d2-16259a98f496.png)

#### [On Classification of (static) MNIST Task]  
about 80% accuracy succeed.

### Temporal Feature
#### [On one pattern prediction]![image]
![one](https://user-images.githubusercontent.com/20160685/123117681-ee102200-d47c-11eb-8770-b16a066e2295.png)
The trained model predicts right-going ball with activations indicating different sparsely represented numbers, so distinguishes the pattern from down-going state. 

## On working theme 
1) One multi pattern prediction
2) On Prediction with (unintentional) saccade
3) On Classification of MNIST with Saccade
