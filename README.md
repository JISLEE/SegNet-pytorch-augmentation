# SegNet-pytorch-augmentation



This repository is based on https://github.com/say4n/pytorch-segnet


![segnet_aug](https://user-images.githubusercontent.com/49095563/121896929-cfaf7580-cd5c-11eb-810f-1a789a6f621c.png)


This is an implementaion of data augmentation including flip and random crop. In order to solve the data imbalance problem of Pascal VOC 2012 dataset, I added thresholding to filter out a background pixel dominent images. 



# Results

![quality](https://user-images.githubusercontent.com/49095563/121897392-48163680-cd5d-11eb-801a-f736c70c811b.png)


Set threshold to 0.5 and aug_num to 1 to get the third result in the above results. 
(Default setting is threshold: 0.65, aug_num: 4)

Training this model takes about 2-3 days. It depends on the threshold and aug_num values. 
