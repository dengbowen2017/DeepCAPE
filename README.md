# DeepCAPE
My own implementation of *DeepCAPE:A Deep Convolutional Neural Network for the Accurate Prediction of Enhancers*

# Preparation 
1. Please download the data from [here](http://health.tsinghua.edu.cn/openness/anno/info/demos/RegulatoryMechanism/data.tar.gz)
2. Make a new folder "raw_data" under "DeepCAPE"
3. Extract the downloaded file and move the content to "raw_data"
4. The file structure will be looked like this

```
└─DeepCAPE
    │  .gitignore 
    │  keras_model.py
    |  keras_train.py
    │  preprocess.py 
    │  pytorch_model.py
    |  pytorch_train.py
    │  README.md 
    │      
    └─raw_data 
        │  epithelial_cell_of_esophagus_differentially_expressed_enhancers.bed 
        │  epithelial_cell_of_esophagus_enhancers.bed 
        │  res_complement.bed 
        │  
        └─Chromosomes 
```

# Requirements
- numpy
- hickle
- keras3
- tensorflow
- pytorch
- sklearn

# How to use
For now, there is something wrong with the pytorch version. Please use keras version instead.
Run *keras_train.py* and you will get the results.

# References
Chen, Shengquan, Mingxin Gan, Hairong Lv, and Rui Jiang. "DeepCAPE: a deep convolutional neural network for the accurate prediction of enhancers." Genomics, Proteomics & Bioinformatics (2021).