# DCASE 2018 Task 3 Bird Audio Detection

This code applies a convolutional neural network (CNN) for bird audio detection. The target is to predict 1 and 0 for the presence and the absence of a bird in an 10 s audio clip, respectively. The code is implemented with PyTorch. 

## Dataset
http://dcase.community/challenge2018/task-bird-audio-detection
<pre>
                 Total	Has bird
--------------------------------------
BirdVoxDCASE20k  20000	10017
freefield1010    7690	1935
warblrb          8000	6045
--------------------------------------
Total            35690	17997

</pre>

## To run
1. Download dataset

2. Modify the dataset path and workspace path in runme.sh

3. Execute command lines in runme.sh line by line. 

## Results
Training takes ~30 ms / iteration on a GeForce GTX 1080 Ti. 

After 3000 iterations, you may get results like:
<pre>
        Error	AUC
----------------------
Train   0.034	0.997
Test    0.152	0.917
</pre>
