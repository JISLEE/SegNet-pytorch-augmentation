from __future__ import print_function
import argparse
from dataset import PascalVOCDataset, NUM_CLASSES
from model import SegNet
import os
import time
import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter


# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = NUM_CLASSES
MOMENTUM = 0.9


NUM_EPOCHS = 2000 
LEARNING_RATE = 0.001
BATCH_SIZE = 32


# Arguments
parser = argparse.ArgumentParser(description='Train a SegNet model')

parser.add_argument('--data_root', required=True)
parser.add_argument('--train_path', required=True)
parser.add_argument('--val_path', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--mask_dir', required=True)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--checkpoint')
parser.add_argument('--gpu', type=int)

args = parser.parse_args()


def eval_net(dataloader):

    correct = 0
    total = 0
    total_pixel = 0
    total_loss = 0
    model.eval() 
    with torch.no_grad():
        for batch in dataloader:
            input_tensor = torch.autograd.Variable(batch['image'])
            target_tensor = torch.autograd.Variable(batch['mask'])
            if CUDA:
                input_tensor = input_tensor.cuda(GPU_ID)
                target_tensor = target_tensor.cuda(GPU_ID)
            predicted_tensor, softmaxed_tensor = model(input_tensor)
            predicted_tensor2 = softmaxed_tensor.argmax(1)
            total += target_tensor.size(0)
            total_pixel += len(target_tensor.flatten())
            correct += (target_tensor == predicted_tensor2).sum()
            loss = criterion(predicted_tensor, target_tensor)
            total_loss += loss.float()
    model.train()
    return total_loss / total, correct.float() / total_pixel



def train():
    is_better = True
    prev_loss = float('inf')
    prev_acc = 0
    
    # For visualize the plots
    writer_train = SummaryWriter(log_dir='./log/train')
    writer_val = SummaryWriter(log_dir='./log/val')

    model.train()
    total_time = 0
    
    for epoch in range(NUM_EPOCHS):
        loss_f = 0
        t_start = time.time()
        total_pixel = 0
        total = 0
        correct = 0
        # bc = 0
        
        for batch in train_dataloader:
            # print("epoch{}, batch{}".format(epoch, bc))
            # bc+=1
            input_tensor = torch.autograd.Variable(batch['image'])
            target_tensor = torch.autograd.Variable(batch['mask'])

            if CUDA:
                input_tensor = input_tensor.cuda(GPU_ID)
                target_tensor = target_tensor.cuda(GPU_ID)

            predicted_tensor, softmaxed_tensor = model(input_tensor)

            optimizer.zero_grad()
            loss = criterion(predicted_tensor, target_tensor)
            loss.backward()
            optimizer.step()


            loss_f += loss.float()
            prediction_f = softmaxed_tensor.float()

        
        delta = time.time() - t_start
        total_time +=delta
        
        print('    Finish training this EPOCH, start evaluating...')
        
        train_loss, train_acc = eval_net(train_dataloader)
        val_loss, val_acc = eval_net(val_dataloader)
        
        # loss plot
        writer_train.add_scalar('augmodel/loss', train_loss, epoch+1)
        writer_val.add_scalar('augmodel/loss', val_loss, epoch+1) 
        # # accuracy plot
        writer_train.add_scalar('augmodel/acc', train_acc, epoch+1)
        writer_val.add_scalar('augmodel/acc', val_acc, epoch+1) 
        
        is_better = prev_acc < val_acc
        is_better2 = train_loss < prev_loss
        
        if is_better or is_better2 or (epoch%500==0):
            prev_acc = val_acc
            prev_loss = train_loss
            saved_dir = os.path.join(args.save_dir, "model_"+str(LEARNING_RATE)+"_"+str(BATCH_SIZE)+"_epoch"+str(epoch+1)+".pth")
            torch.save(model.state_dict(), saved_dir)
            print("got better model, saved to {} (acc: {:.8f})".format(saved_dir, val_acc))
            

        print("Epoch #{} Train Loss: {:.8f} Train Acc: {:.8f} Val Loss: {:.8f} Val Acc: {:.8f} Time: {:2f}s".format(epoch+1, train_loss, train_acc, val_loss, val_acc, delta))
    print("total time:", total_time)
    writer_train.close()
    writer_val.close()


if __name__ == "__main__":
    data_root = args.data_root
    train_path = os.path.join(data_root, args.train_path)
    val_path = os.path.join(data_root, args.val_path)
    img_dir = os.path.join(data_root, args.img_dir)
    mask_dir = os.path.join(data_root, args.mask_dir)

    CUDA = args.gpu is not None
    GPU_ID = args.gpu


    train_dataset = PascalVOCDataset(list_file=train_path,
                                     img_dir=img_dir,
                                     mask_dir=mask_dir, 
                                     tag=1)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=4)
    
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

        # class_weights = 1.0/train_dataset.get_class_probability().cuda(GPU_ID)
        # criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(GPU_ID)
        criterion = torch.nn.CrossEntropyLoss().cuda(GPU_ID) # train with pretrained model

    else:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS)

        # class_weights = 1.0/train_dataset.get_class_probability()
        # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        criterion = torch.nn.CrossEntropyLoss()


    if args.checkpoint:
        print("load checkpoint:{}".format(args.checkpoint))
        # model.load_state_dict(torch.load(args.checkpoint))
        model.load_state_dict(torch.load(SAVED_MODEL_PATH, 'cuda:'+str(GPU_ID)))


    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=LEARNING_RATE)


    train()
