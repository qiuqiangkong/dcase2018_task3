# DCASE 2018 Task 3 Bird audio detection

DCASE 2018 Task 3 bird audio detection is a challenge to detect the presence or the absence of birds in 10 second audio clips. We provide a convolutional neural network (CNN) baseline system implemented with PyTorch in this code base. More details about this challenge can be found in http://dcase.community/challenge2018/task-bird-audio-detection

## DATASET

The dataset is downloadable from http://dcase.community/challenge2018/task-bird-audio-detection

The development data consists of audio clips from freefield1010, warblrb10k and BirdVox-DCASE-20k. The evaluation data consists of audio clips from warblrb10k, Chernobyl and PolandNFC dataset. 

### Statistics of development data

| Dataset name      | Number of audio clips |
|-------------------|:---------------------:|
| freefield1010     |          7690         |
| warblrb10k        |          8000         |
| BirdVox-DCASE-20K |         20000         |
| **Total**         |         **35690**     |

### Statistics of test data

| Dataset name | Number of audio clips |
|--------------|:---------------------:|
| warblrb10k   |          2000         |
| Chernobyl    |          6620         |
| PolandNFC    |          4000         |
| **Total**    |         **12620**     |


The log mel spectrogram of the scenes are shown below:

![alt text](appendixes/logmel.png)

## Run the code
**1. (Optional) Install dependent packages.** If you are using conda, simply run:

$ conda env create -f environment.yml

$ conda activate py3_dcase2018_task1

**2. Then simply run:**

$ ./runme.sh

Or run the commands in runme.sh line by line, including: 

(1) Modify the paths of data and your workspace

(2) Extract features

(3) Train model

(4) Evaluation

The training looks like:

<pre>
root        : INFO     Loading data time: 7.601605415344238
root        : INFO     Split development data to 6122 training and 2518 validation data. 
root        : INFO     Number of train audios in specific devices ['a']: 6122
root        : INFO     tr_acc: 0.100
root        : INFO     Number of validate audios in specific devices ['a']: 2518
root        : INFO     va_acc: 0.100
root        : INFO     iteration: 0, train time: 0.006 s, validate time: 2.107 s
root        : INFO     ------------------------------------
......
root        : INFO     Number of train audios in specific devices ['a']: 6122
root        : INFO     tr_acc: 1.000
root        : INFO     Number of validate audios in specific devices ['a']: 2518
root        : INFO     va_acc: 0.688
root        : INFO     iteration: 3000, train time: 6.966 s, validate time: 2.340 s
root        : INFO     ------------------------------------
root        : INFO     Number of train audios in specific devices ['a']: 6122
root        : INFO     tr_acc: 1.000
root        : INFO     Number of validate audios in specific devices ['a']: 2518
root        : INFO     va_acc: 0.688
root        : INFO     iteration: 3100, train time: 6.266 s, validate time: 2.345 s
</pre>

## Result

We apply a convolutional neural network on the log mel spectrogram feature to solve this task. Training takes around 100 ms / iteration on a GTX Titan X GPU. The model is trained for 3000 iterations. The result is shown below. 

### Subtask A

Averaged accuracy over 10 classes:

|                   | Device A |
|:-----------------:|:--------:|
| averaged accuracy |   68.2%  |

Confusion matrix:

<img src="https://github.com/qiuqiangkong/dcase2018_task1/blob/dev/appendixes/subtask_a_confusion_matrix.png" width="600">

### Subtask B

Averaged accuracy over 10 classes of device A, B and C:

|                   | Device A | Device B | Device C |
|:-----------------:|:--------:|----------|----------|
| averaged accuracy |   67.4%  | 59.4%    | 57.2%    |

Confusion matrix:

<img src="https://github.com/qiuqiangkong/dcase2018_task1/blob/dev/appendixes/subtask_b_confusion_matrix_device_a.png" width="400"><img src="https://github.com/qiuqiangkong/dcase2018_task1/blob/dev/appendixes/subtask_b_confusion_matrix_device_b.png" width="400">
<img src="https://github.com/qiuqiangkong/dcase2018_task1/blob/dev/appendixes/subtask_b_confusion_matrix_device_c.png" width="400">

## Summary
This codebase provides a convolutional neural network (CNN) for DCASE 2018 challenge Task 1. 

### External link

The official baseline system implemented using Keras can be found in https://github.com/DCASE-REPO/dcase2018_baseline
