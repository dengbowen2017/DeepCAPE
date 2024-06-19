# DeepCAPE
My own implementation of *DeepCAPE:A Deep Convolutional Neural Network for the Accurate Prediction of Enhancers*

# Preparation 
1. Please download the data from [here](http://health.tsinghua.edu.cn/openness/anno/info/demos/RegulatoryMechanism/data.tar.gz)
2. Make a new folder "raw_data" under "DeepCAPE"
3. extract the downloaded file and move the content to "raw_data"
4. the file structure will be looked like this

```
└─DeepCAPE
    │  .gitignore 
    │  dataset_and_model.py 
    │  preprocess.py 
    │  README.md 
    │  train.py 
    │      
    └─raw_data 
        │  epithelial_cell_of_esophagus_differentially_expressed_enhancers.bed 
        │  epithelial_cell_of_esophagus_enhancers.bed 
        │  res_complement.bed 
        │  
        └─Chromosomes 
```

# References
Chen, Shengquan, Mingxin Gan, Hairong Lv, and Rui Jiang. "DeepCAPE: a deep convolutional neural network for the accurate prediction of enhancers." Genomics, Proteomics & Bioinformatics (2021).