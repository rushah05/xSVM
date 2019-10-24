# xSVM
xSVM is a fast and scalable SVM classifier.

## Required modules
- gcc 7.1.0  
- mkl 18.0.2  
- export julia 1.0.4  

## Parameters
n - no of records  
d - no of features  
k - Kernel rank  
C -    
g - gamma  
q - no of iterations to orthogonalize the approximated kernel matrix (for better accuracy)  

## How to train
We are performing parallelization using MPI to train the classifier on a cluster

Sample cmdline:  
time mpirun xSVM <data set file path> n d k g q C

## How to predict
We are using our own serial prediction code written in Julia to predict

Sample cmdline:   
time julia ../julia/predict.jl <model file name> <data set file path> n
