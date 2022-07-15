

# 2D Object detection

Waymo Data set was used for the 2D Object detection.
Tensorflow Object detection (TFOD) was used to train and evaluate.
ssd_resnet50_v1_fpn_640x640_coco17 pretrained model weights were used to train.

The EDA, training and evaluation was done on google colab pro.
The notebooks have been provided on the git repo [2d Object detection](https://github.com/pavanp5/2d_object_detection).

The metadata such as timeofday, location and weather was extracted from the waymo data ,while converting to tfrecord format suitable for TFOD object detection model.
This metadata has been used to perform the train & eval split. The CSV files with metadata have been provided here [train_eval_split](https://github.com/pavanp5/2d_object_detection/tree/main/data)

The train set included data that was very representative. This can be evidenced in the [EDA notebook](https://github.com/pavanp5/2d_object_detection/blob/main/EDA_Train_Eval_Split.ipynb).

4 experiments were performed in total. 3 of the experiments yielded improvement in precision and accuracy. The evaluation metrics and model visualization of detections are available here [Eval metrics notebook](https://github.com/pavanp5/2d_object_detection/blob/main/Evaliation_&_Model_Result_Visualization.ipynb).

|Model#  | Eval mAP@50 |Eval AR@100  | Eval total loss |Technique|
|--|--|--|--|--|
|  Exp1|28.82%  | 21.98% | 0.7441 |No Augmentation|
|Exp2| 28.9% | 21.3%  |  0.7386|Augmentation random crop, random contrast adjust
|Exp4|29.03|21.33%|0.7091|Sklearn Kmeans on gt boxes to get anchor aspect ratios, Scales per octave increased to 3, Augmentation-random crop,random contrast adjust

**Model visualization:**
Exp4 detection images and videos:
![SFO day time Detections](./images/sfo_day.png)
![Phoenix rainy night Detections](./images/phx_rain.png)
![SFO night Detections](./images/sfo_night.png)
![SFO rainy day Detections](./images/sfo_rain_day.png)

**Videos of detections:**
[Day time detection video, ](https://github.com/pavanp5/2d_object_detection/blob/main/videos/animation_eval_otherloc_day.mp4)
[Phoenix rain video, ](https://github.com/pavanp5/2d_object_detection/blob/main/videos/animation_eval_phx_rain.mp4)
[SFO Dawn/Dusk detection video, ](https://github.com/pavanp5/2d_object_detection/blob/main/videos/animation_eval_sfo_dawndusk.mp4)
[SFO Day time detection video, ](https://github.com/pavanp5/2d_object_detection/blob/main/videos/animation_eval_sfo_day.mp4)
[SFO Night time detection video](https://github.com/pavanp5/2d_object_detection/blob/main/videos/animation_eval_sfo_night.mp4)

**What Augmentations were used and why?**
The exp1 model with no augmentation overfits. The number of images are considerably high but they are from same sequences. 202 sequences were used for training each having 200 images approximately.  So, random crop was used so that the images of the sequence are cropped and at different regions thus reducing overfit. The second augmentation used was random adjust contrast. The data has night sequences and dawn/dusk sequences, so adjusting contrast would help train for the dawn/ dusk scenarios too. The augmentations can be viewed here [Augmentations notebook](https://github.com/pavanp5/2d_object_detection/blob/main/Augmentations.ipynb).

**What changes were made to improve model over after augmentation was applied?**
Exp4 config metric improved, the eval loss reduced compared to previous experiments. 
2 Changes below were made
 1. Anchor box aspect ratios were calculated from train dataset ground truth boxes. 
 2. Scales per octave was increased to 3

Kmeans Clustering was used on the ground truth bounding boxes to derive suitable anchor box aspect ratios. The code and plots for the same can be observed at the end of this notebook [anchor box calculation at the end of this notebook](https://github.com/pavanp5/2d_object_detection/blob/main/Augmentations.ipynb).

**Other details:**
Cosine decay scheduler has been used. The classification, localization loss and learning rate decay plots can be viewed on tensor board plots.

**Training Experiment4** 
![Exp4 Tensor board](./images/exp4_tfboard_train.PNG)

**Training Experiment3**
![Exp3 Tensor board](./images/exp3_tfboard_train.PNG)

**Training Experiment1**
![Exp1 Tensor board](./images/exp1_tfboard_train.PNG)

**Evaluation metric Experiment4**
![Eval metric Experiment4 ](./images/Exp4_eval_metric.PNG)

**Evaluation metric Experiment3**
![Eval metric Experiment3](./images/Exp3_eval_metric.PNG)

**Evaluation metric Experiment1**
![Eval metric Experiment1](./images/Exp1_eval_metric.PNG)

**Category wise metric for experiment 4 (Cars, Pedestrian and Cyclist):**
**Precision Category level Exp4**
![category metric precision exp4](./images/category_metric_precision.PNG)
**Recall category level Exp 4**
![category metric recall exp4](./images/category_metric_recall.PNG)

Sigmoid Focal loss was used for classification loss to have the hard examples learnt given the imbalance between classes. Gamma value was increased greater than 2 but was suitable. Gamma equal zero is equivalent to categorical cross entropy and was not suitable.

Based on the study or referring of few related research papers the following will be taken up as next steps, i.e. building the model from scratch thereby having control to extract features from layers and fuse them. Currently SSD is not very good with small objects. This is because the lower layers which are used for small objects detection do not have enough information to detect object. If information from other layers higher up can be fused with this the prediction small objects can be improved. 