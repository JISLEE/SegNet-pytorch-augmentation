from collections import Counter
# import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

VOC_CLASSES = ('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

NUM_CLASSES = len(VOC_CLASSES) + 1 #22

class PascalVOCDataset(Dataset):
    """Pascal VOC 2007 Dataset"""
    def __init__(self, list_file, img_dir, mask_dir, transform=None, tag=1):
        self.images = open(list_file, "rt").read().split("\n")[:-1]
        self.transform = transform

        self.img_extension = ".jpg"
        self.mask_extension = ".png"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir
        
        # only augment data for train
        self.tag = tag
        self.aug_num = 4
        self.data_num = len(self.images)

        if self.tag==1:
            self.x, self.y = self.data_aug() 
            self.counts = self.__compute_class_probability_aug() #consider augmented dataset

        if self.tag==0:
            self.counts = self.__compute_class_probability()
        
    def __len__(self):
        return self.data_num 

    def __getitem__(self, index):
        
        # data augmentation
        if self.tag==1: # for training set
            data = {
                    'image': torch.FloatTensor(self.x[index].copy()),
                    'mask' : torch.LongTensor(self.y[index].copy())
            }
            # print(data)
            return data

        elif self.tag==0: #for validation set
            name = self.images[index]
            image_path = os.path.join(self.image_root_dir, name + self.img_extension)
            mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)
            image = self.load_image(path=image_path) 
            gt_mask = self.load_mask(path=mask_path)
            data = {
                    'image': torch.FloatTensor(image),
                    'mask' : torch.LongTensor(gt_mask)
                    }
            return data
       
        

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))

        for name in self.images:
            mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

            raw_image = Image.open(mask_path).resize((224, 224))
            imx_t = np.array(raw_image).reshape(224*224)
            imx_t[imx_t==255] = len(VOC_CLASSES)

            for i in range(NUM_CLASSES):
                counts[i] += np.sum(imx_t == i)

        return counts
    
    def __compute_class_probability_aug(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))
        mask_list = self.y
        for mask in mask_list: #mask: (224,224) np


            for i in range(NUM_CLASSES):
                counts[i] += np.sum(mask == i)
        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values/np.sum(values)

        return torch.Tensor(p_values)

    def load_image(self, path=None):
        raw_image = Image.open(path)
        raw_image = np.transpose(raw_image.resize((224, 224)), (2,1,0))
        imx_t = np.array(raw_image, dtype=np.float32)/255.0

        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize((224, 224))
        imx_t = np.array(raw_image)
        # border
        imx_t[imx_t==255] = len(VOC_CLASSES)

        return imx_t
    
    def data_aug(self):

        imgs = []
        masks = []
        total_data = 0
        
        print("data augmentation -----")

        
        for name in self.images:
            ipath = os.path.join(self.image_root_dir, name + self.img_extension)
            mpath = os.path.join(self.mask_root_dir, name + self.mask_extension)
            
            img = Image.open(ipath) # (hight, width, channel) #PIL
            mask = Image.open(mpath) # (hight, width) #PIL
            
            h =  np.array(img).shape[0]
            w =  np.array(img).shape[1]
            
            #resize original and decide whether to append or not
            img_resize = img.resize((224,224)) #PIL
            mask_resize = mask.resize((224,224)) #PIL
            np_mask = np.array(mask_resize) #np
            # border 255 to 21
            np_mask[np_mask==255] = len(VOC_CLASSES) #np
            
            o_ratio = ((np_mask==0).sum())/(224*224)
            if o_ratio<0.65:
                img_resize = np.transpose(img_resize, (2,1,0)) #PIL
                img_resize = np.array(img_resize, dtype=np.float32)/255.0 #np
                
                imgs.append(img_resize)
                masks.append(np_mask)
                total_data +=1
                
                # flip up-down and left-right
                imgs.append(np.flipud(img_resize)) #np
                masks.append(np.flipud(np_mask))
                total_data +=1
                
                imgs.append(np.fliplr(img_resize)) #np
                masks.append(np.fliplr(np_mask))
                total_data +=1

            # crop images and masks
            if h>224 and w>224:
                for i in range(self.aug_num):
                    randw = np.random.randint(w-224)
                    randh = np.random.randint(h-224)

                    img_crop = img.crop((randh, randw, randh+224, randw+224)) #PIL
                    mask_crop = mask.crop((randh, randw, randh+224, randw+224)) #PIL
                    mask_crop = np.array(mask_crop) #np
                    # border 255 to 21
                    mask_crop[mask_crop==255] = len(VOC_CLASSES) #np
                    

                    bg_ratio = ((mask_crop==0).sum())/(224*224)
                    if bg_ratio<0.65:
                        img_cropt = np.transpose(img_crop, (2,1,0)) #PIL
                        img_crop_np = np.array(img_cropt, dtype=np.float32)/255.0 #np

                        imgs.append(img_crop_np)
                        masks.append(mask_crop)
                        total_data +=1
                        
                        # flip up-down and left-right
                        imgs.append(np.flipud(img_crop_np)) #np
                        masks.append(np.flipud(mask_crop))
                        total_data +=1

                        imgs.append(np.fliplr(img_crop_np)) #np
                        masks.append(np.fliplr(mask_crop))
                        total_data +=1
        print("data augment number:", total_data)
        self.data_num = len(imgs)
        
        # print(len(imgs))

                        
        return imgs, masks
        
        


if __name__ == "__main__":
    data_root = os.path.join("data", "VOCdevkit", "VOC2007")
    list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
    img_dir = os.path.join(data_root, "JPEGImages")
    mask_dir = os.path.join(data_root, "SegmentationObject")


    objects_dataset = PascalVOCDataset(list_file=list_file_path,
                                       img_dir=img_dir,
                                       mask_dir=mask_dir)

    print(objects_dataset.get_class_probability())

    sample = objects_dataset[0]
    image, mask = sample['image'], sample['mask']

    image.transpose_(0, 2)

    # fig = plt.figure()

    # a = fig.add_subplot(1,2,1)
    # plt.imshow(image)

    # a = fig.add_subplot(1,2,2)
    # plt.imshow(mask)

    # plt.show()

