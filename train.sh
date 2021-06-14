CUDA_VISIBLE_DEVICES=0,1 python train.py --data_root /nfs/hpc/share/leejis/data/VOCdevkit/VOC2012/ \
                --train_path ImageSets/Segmentation/train.txt \
                --val_path ImageSets/Segmentation/val.txt \
                --img_dir JPEGImages \
                --mask_dir SegmentationClass \
                --save_dir /nfs/hpc/share/leejis/pySegNet/saved_models/\
                --gpu 0

