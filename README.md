# DADM_NeuralNetworkProject_2018-2019
## Contributors
NeuralNetworkProject has been developed by Francesco Musso ([@frmusso](https://github.com/frmusso)) and Davide Ponassi ([@ponassi](https://github.com/ponassi)).

## The Aim
Automated methods to detect and classify human diseases from medical images.
In this project we are dealing with classifying if a patient is affected or not with pneumonia from chest x-ray images.

## The Dataset
The dataset is originally organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal).
We decided to reorganize the dataset into two folders (Pneumonia/Normal) and divide the dataset into trainSet, validationSet and testSet by code. 
There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

## The Structure
To see the differences in performance and accuracy, we will make two different neural network.
- Single perceptron (implemented using the PLA)
- Neural net with one hidden layer of four neuron and one output layer of a single neuron (implemented using the Backpropagation)

## Single Perceptron
Each x-ray is set to greyscale and reshaped to 100x100 and then standardized.
The dataset is divided with a proportion of 60:20:20 (trainingSet:validationSet:testSet).
The perceptron learn using the PLA.
We use the sign as loss function.
At the end we print the final weight learnt by the perceprtron.

<p align="center">
  <img width="460" height="300" src="https://github.com/ponassi/DA-DM_NeuralNetworkProject_2018-2019/blob/master/Perceptron/perceptron_final_weights.png">
</p>

## Neural Net
Each x-ray is set to greyscale and reshaped to 100x100 and then normalized.
The dataset is divided with a proportion of 60:20:20 (trainingSet:validationSet:testSet).
The NN learn using Batch Gradient Descent and Backpropagation.
We use the MSE as loss function.

<p align="center">
<img src="https://github.com/ponassi/DA-DM_NeuralNetworkProject_2018-2019/blob/master/NeuralNet/mom_acc_25000plus.PNG" width="425" height="350"/> 
<img src="https://github.com/ponassi/DA-DM_NeuralNetworkProject_2018-2019/blob/master/NeuralNet/mom_loss_25000plus.PNG" width="425" height="350"/> 
</p>
