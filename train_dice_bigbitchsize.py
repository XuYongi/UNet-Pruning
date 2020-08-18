import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net,gene_eval_data
# from unet import UNet
# from unet.model import Modified2DUNet as UNet
# from unet.model_level3_withsigmoidfordiceloss import Modified2DUNet as UNet
from unet.model_level3_dl_dilationCov_transConv import Modified2DUNet as UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from dice_loss import dice_loss
from focal_loss import FocalLoss
# dir_img = '../data/train/resized_BG/imgs/'
# dir_mask = '../data/train/resized_BG/masks/'  #使用膨胀后的标签

dir_img = './data/train/imgs/'
dir_mask = './data/train/masks/'  #使用膨胀后的标签
# dir_img = '../data/train/test/imgs/'
# dir_mask = '../data/train/test/imgs/'  #使用膨胀后的标签

dir_checkpoint = './checkpoints/'

def train_net(net,
              device,
              epochs=100,
              batch_size=1,
              lr=0.1,
              val_percent=0.2,
              save_cp=True,
              img_scale=1):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=16, pin_memory=True,drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False,  num_workers=8, pin_memory=True, drop_last=True)

    gene_eval_data(val_loader, dir='./data/val/')

    writer = SummaryWriter(comment='LR_{}_BS_{}_SCALE_{}'.format(lr,batch_size,img_scale))
    global_step = 0

    logging.info('''Starting training:
        Epochs:          {}
        Batch size:      {}
        Learning rate:   {}
        Training size:   {}
        Validation size: {}
        Checkpoints:     {}
        Device:          {}
        Images scaling:  {}
    '''.format(epochs,batch_size,lr,n_train,n_val,save_cp,device.type,img_scale))

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max',factor=0.5, patience=20)

    criterion = dice_loss
    # criterion = nn.BCELoss()

    last_loss =9999
    last_val_score = 0
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        step = 0
        mybatch_size = 4
        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch + 1,epochs), unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels,\
                    'Network has been defined with {} input channels, '.format(net.n_channels)+\
                'but loaded images have {} channels. Please check that '.format(imgs.shape[1])+\
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                global_step += 1
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                step +=1
                if step% mybatch_size ==0:

                    optimizer.step()
                    optimizer.zero_grad()
                    step = 0

                pbar.update(imgs.shape[0])


# if global_step % (len(dataset) // ( 2* batch_size)) == 0:
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
        val_score = eval_net(net, val_loader, device)
        scheduler.step(val_score)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        if net.n_classes > 1:
            logging.info('Validation cross entropy: {}'.format(val_score))
            writer.add_scalar('Loss/test', val_score, global_step)
        else:
            logging.info('Train Loss: {}    Validation Dice Coeff: {} '.format(epoch_loss/n_train , val_score))
            writer.add_scalar('Dice/test', val_score, global_step)

            writer.add_images('images', imgs, global_step)
            if net.n_classes == 1:
                writer.add_images('masks/true', true_masks, global_step)
                writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.3 , global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if last_loss > epoch_loss or last_val_score < val_score:
                last_loss  = min (last_loss, epoch_loss)
                last_val_score = max(last_val_score , val_score)
                # torch.save(net.state_dict(),
                torch.save(net,
                           dir_checkpoint + 'CP_epoch{}Trainloss{}ValDice{}.pt'.format(epoch + 1,epoch_loss/n_train, val_score))
                logging.info('Checkpoint {} saved !'.format(epoch + 1)+'   CP_epoch{}Trainloss{}ValDice{}.pt'.format(epoch + 1,epoch_loss/n_train, val_score))

    writer.close()

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        # nn.init.constant_(m.bias.data, 0.0)   #有BN层不需要bias
    # elif classname.find('ConvTranspose2d') != -1:
    #     nn.init.xavier_normal_(m.weight.data)
    #     nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias, 0.0)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=500,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0025,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,#'./checkpoints/CP_epoch31Trainloss0.17970650666497748ValDice0.8402551859617233.pth',
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    #torch.cuda.set_device(0)
    #torch.set_num_threads(1)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {}'.format(device))

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=1)
    if True:
        logging.info('Using init {}'.format(device))
        net.apply(weights_init)
    logging.info('Network:\n'
                 +'\t{} input channels\n'.format(net.n_channels)+
                 '\t{} output channels (classes)\n'.format(net.n_classes)+
                 '\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
