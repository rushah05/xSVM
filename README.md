# xSVM
xSVM is a fast and scalable SVM classifier.
Please refer the below link to access the full paper - https://ieeexplore.ieee.org/document/9006315

## Required modules
- gcc 7.1.0  
- mkl 18.0.2  
- export julia 1.0.4  

## Parameters
n - no of records  
d - no of features  
k - kernel rank  
C - regularization parameter   
g - gamma  
q - no of iterations to orthogonalize the approximated kernel matrix (for better accuracy)  
mdfile - model.csv file created after training  
dspath - data set file path

## How to train
We are performing parallelization using MPI to train the classifier on a cluster

Sample cmdline:  
time mpirun xSVM dspath n d k g q C

## How to predict
We are using our own serial prediction code written in Julia to predict

Sample cmdline:   
time julia ../julia/predict.jl mdfile dspath n
