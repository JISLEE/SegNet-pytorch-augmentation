from __future__ import print_function
import argparse
from dataset import PascalVOCDataset, NUM_CLASSES
import matplotlib.pyplot as plt
from model import SegNet
import numpy as np
import os
from PIL import Image
import time
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from metrics import AverageMeter, IOUMetric


plt.switch_backend('agg')
plt.axis('off')


# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = NUM_CLASSES

BATCH_SIZE = 32


# Arguments
parser = argparse.ArgumentParser(description='Validate a SegNet model')

parser.add_argument('--data_root', required=True)
parser.add_argument('--val_path', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--mask_dir', required=True)
parser.add_argument('--model_path', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--gpu', type=int)


args = parser.parse_args()


cls_invert = {0:(0,0,0),1:(128,0,0),2:(0,128,0), # 0:background, 1:aeroplane, 2:bicycle
               3:(128,128,0),4:(0,0,128),5:(128,0,128), #3:bird, 4:boat, 5:bottle
               6:(0,128,128),7:(128,128,128),8:(64,0,0), #6:bus, 7:car, 8:cat
               9:(192,0,0),10:(64,128,0),11:(192,128,0), #9:chair, 10:cow, 11:diningtable
               12:(64,0,128),13:(192,0,128),14:(64,128,128), #12:dog, 13:horse, 14:motorbike
               15:(192,128,128),16:(0,64,0),17:(128,64,0), #15:person, 16:pottedplant, 17:sheep
               18:(0,192,0),19:(128,192,0),20:(0,64,128), #18:sofa, 19:train, 20:tvmonitor
               21:(224, 224, 192)} #21:default

def save_res(input_tensor, target_tensor, softmaxed_tensor, batch_idx):
    
    for idx, predicted_mask in enumerate(softmaxed_tensor):
                
        target_mask = target_tensor[idx] # tensor (224,224)
        input_image = input_tensor[idx]
        
        fig = plt.figure()

        a = fig.add_subplot(1,3,1)
        plt.imshow(input_image.transpose(0, 2).cpu())
        a.set_title('Input Image')

        a = fig.add_subplot(1,3,2)
        predicted_mx = predicted_mask.detach().cpu().numpy() #(22, 224, 224)
        predicted_mx = predicted_mx.argmax(axis=0) #(224,224)
        predicted_rgb = torch.zeros_like(input_image.transpose(0, 2)) #(224,224,3)

        for i in range(predicted_rgb.shape[0]):
            for j in range(predicted_rgb.shape[1]):
                predicted_rgb[i,j] = torch.IntTensor(cls_invert[predicted_mx[i][j]])
        predicted_rgb = predicted_rgb/255        
        
        plt.imshow(predicted_rgb.cpu())
        a.set_title('Predicted Mask')

        a = fig.add_subplot(1,3,3)
        target_mx = target_mask.detach().cpu().numpy() #(224,224)
        mask_rgb = torch.zeros_like(input_image.transpose(0, 2)) #(224,224, 3)

        for i in range(mask_rgb.shape[0]):
            for j in range(mask_rgb.shape[1]):
                mask_rgb[i,j] = torch.IntTensor(cls_invert[target_mx[i][j]])
        mask_rgb = mask_rgb / 255
        plt.imshow(mask_rgb.cpu())
        a.set_title('Ground Truth')
        
        fig.savefig(os.path.join(OUTPUT_DIR, "result_{}_{}.png".format(batch_idx, idx)))

        plt.close(fig)

        # if batch_idx==1:
            #     break


def validate():
    model.eval()
    
    t_start = time.time()
    correct = 0
    total = 0
    total_pixel = 0

    inter = np.zeros(NUM_CLASSES)
    iou = np.zeros(NUM_CLASSES)
    count = np.zeros(NUM_CLASSES)

    metrics = IOUMetric(NUM_CLASSES)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            input_tensor = torch.autograd.Variable(batch['image'])
            target_tensor = torch.autograd.Variable(batch['mask'])

            print("batch {}".format(batch_idx))
            if CUDA:
                input_tensor = input_tensor.cuda(GPU_ID)
                target_tensor = target_tensor.cuda(GPU_ID)

            predicted_tensor, softmaxed_tensor = model(input_tensor)
            predicted_tensor2 = softmaxed_tensor.argmax(1)

            # pixel accuracy
            total_pixel += len(target_tensor.flatten())
            correct += (target_tensor == predicted_tensor2).sum()

            #iou 
            metrics.add_batch(predicted_tensor2.data.cpu().numpy(), target_tensor.data.cpu().numpy())
            
            # save result images
            save_res(input_tensor, target_tensor, softmaxed_tensor, batch_idx)

    pixel_acc = correct.float() / total_pixel
    acc, _, iou_class, mean_iou, _ = metrics.evaluate()
    delta = time.time() - t_start

    print("pixel accuracy:{}".format(acc))
    print("IoU per classes:{}, mIOU:{}".format(iou_class, mean_iou))
    
    print("total inference time: ", delta)


if __name__ == "__main__":
    data_root = args.data_root
    val_path = os.path.join(data_root, args.val_path)
    img_dir = os.path.join(data_root, args.img_dir)
    mask_dir = os.path.join(data_root, args.mask_dir)

    SAVED_MODEL_PATH = args.model_path
    OUTPUT_DIR = args.output_dir

    CUDA = args.gpu is not None
    GPU_ID = args.gpu


    val_dataset = PascalVOCDataset(list_file=val_path,
                                   img_dir=img_dir,
                                   mask_dir=mask_dir, 
                                   tag=0)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=4)


    if CUDA:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS).cuda(GPU_ID)
    else:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS)

    model.load_state_dict(torch.load(SAVED_MODEL_PATH, 'cuda:'+str(GPU_ID)))

    validate()



