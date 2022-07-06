
# 2D Object detection

Waymo Data set was used for the 2D Object detection.
Tensorflow Object detection (TFOD) was used to train and evaluate.
ssd_resnet50_v1_fpn_640x640_coco17 pretrained model weights were used to train.

The EDA, training and evaluation was done on google colab pro.
The notebooks have been provided on the git repo [2d Object detection](https://github.com/pavanp5/2d_object_detection).

The metadata such as timeofday, location and weather was extracted from the waymo data ,while converting to tfrecord format suitable for TFOD object detection model.
This metadata has been used to perform the train & eval split. The CSV files with metadata have been provided here [train_eval_split](https://github.com/pavanp5/2d_object_detection/tree/main/data)

The train set included data that was very representative. This can be evidenced in the [EDA notebook](https://github.com/pavanp5/2d_object_detection/blob/main/EDA_Train_Eval_Split.ipynb).

9 experiments were performed in total. 4 of the experiments yeilded improvement in precision and accuracy. The metrics have been capture at the class level and are available here [training metrics notebook](https://github.com/pavanp5/2d_object_detection/blob/main/Training_Experiment_Results.ipynb).

Many augmentations were tried of which contrast adjust augmentation yielded an improved precision and recall. The augmentations can be viewed here [Augmentations notebook](https://github.com/pavanp5/2d_object_detection/blob/main/Augmentations.ipynb).

Increasing the scales per octave to 3 improved the precision and recall. But further increase to scales per octave reduced the precision and recall.

nms threshold of 1e-02 and 1e-01 were tried but they reduced the precision and recall and so 1e-08 was used with iou of 0.6.

Kmeans Clustering was used on the ground truth bounding boxes to derive suitable anchor box aspect ratios. The code and plots for the same can be observed at teh end of this notebook [anchor box calculation at the end of this notebook](https://github.com/pavanp5/2d_object_detection/blob/main/Augmentations.ipynb).

Cosine decay scheduler has been used. The classification, localization loss and learning rate decay plots can ve viewed on the tensorboard in this notebook [Tensorboard dev plots](https://github.com/pavanp5/2d_object_detection/blob/main/Tensorboard_log_plots.ipynb).

The class level metrics has been provided in this notebook for the experiments that showed improved precision and recall here [metrics notebook](https://github.com/pavanp5/2d_object_detection/blob/main/Training_Experiment_Results.ipynb).

Sigmoid Focal loss was used for classification loss to have the hard examples learnt given the imbalance between classes. Gamma value was increased greater than 2 but was suitable. Gamma equal zero is equivalent to binary cross entropy and was not suitable.

Based on the study or referring of few related research papers the following will be taken up as next steps, i.e. building the model from scratch thereby having control to extract features from layers and fuse them. Currently SSD is not very good with small objects. This is because the lower layers which are used for small objects detection do not have enough information to detect object. If information from other layers higher up can be fused with this the prediction small objects can be improved. 