
CUDA_VISIBLE_DEVICES=0,1 python inference.py --data_root /nfs/hpc/share/leejis/data/VOCdevkit/VOC2012/ \
                --val_path ImageSets/Segmentation/val.txt \
                --img_dir JPEGImages \
                --mask_dir SegmentationClass \
                --model_path /nfs/hpc/share/leejis/pySegNet/saved_models/model_0.001_32_epoch1001.pth \
                --output_dir /nfs/hpc/share/leejis/pySegNet/output \
                --gpu 0
