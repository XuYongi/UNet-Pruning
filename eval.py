import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from dice_loss import dice_coeff


def mask_to_image(mask):
    return Image.fromarray((mask.numpy() * 255).astype(np.uint8))

def img_to_image(img):
    img = np.transpose(img,[1,2,0])
    return Image.fromarray((img.numpy() * 255).astype(np.uint8))

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    # with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        # if net.n_classes > 1:
        #     tot += F.cross_entropy(mask_pred, true_masks).item()
        # else:
        #     pred = torch.sigmoid(mask_pred)
        pred = (mask_pred > 0.3).float()
        tot += dice_coeff(pred, true_masks).item()
            # pbar.update()

    return tot / n_val

def gene_eval_data( loader, dir = '../data/val/'):
    """Evaluation without the densecrf with the dice coefficient"""

    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    for i,batch in enumerate(loader):
        imgs, true_masks = batch['image'], batch['mask']
        print(imgs.shape)
        if not os.path.exists(dir):
            os.mkdir(dir)
        result = img_to_image(imgs[0])
        result.save(dir+'{}.jpg'.format(i))
    return 0
