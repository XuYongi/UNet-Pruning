import cv2
import glob
import numpy as np
import os
from PIL import Image
imglist  = glob.glob('../data/masks_ori_dilation/*.png')
imglist.sort()
outpath = '../data/masks_resize/'
#
#
imglist1  = glob.glob('../data/imgs_ori/*.jpg')
imglist1.sort()
outpath1 = '../data/imgs_resize/'
#

# imglist  = glob.glob('../data/test/masks/*.png')
# imglist.sort()
# outpath = '../data/test/masks_dilation/'
#

# imglist  = glob.glob('../data/masks_ori/*.png')
# imglist.sort()
# outpath = '../data/masks_ori_dilation/'

def dilation(imglist,outpath):
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    for i in range(len(imglist)):
        print("***",i,'/',len(imglist))

        src = cv2.imread(imglist[i], cv2.IMREAD_UNCHANGED)

        ## b.设置卷积核5*5
        kernel = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((1, 1), np.uint8)

        ## 图像的膨胀
        dst = cv2.dilate(src, kernel)

        # c.图像的腐蚀，默认迭代次数
        erosion = cv2.erode(dst, kernel2)

        cv2.imwrite(outpath+os.path.basename(imglist[i]),erosion)


def resize(imglist, outpath):
    newW, newH = 1440,900
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    for i in range(len(imglist)):
        print("***", i, '/', len(imglist))
        pil_img = Image.open(imglist[i])
        dst = pil_img.resize((newW, newH),Image.ANTIALIAS)
        dst.save(outpath + os.path.basename(imglist[i]))

def resize_mask(imglist, outpath):
    newW, newH = 1440,900
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    for i in range(len(imglist)):
        print("***", i, '/', len(imglist))
        pil_img = Image.open(imglist[i])
        dst = pil_img.resize((newW, newH),Image.NEAREST)
        dst.save(outpath + os.path.basename(imglist[i]))

def xiugai(imglist, outpath):
    newW, newH = 1440,900
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    for i in range(len(imglist)):
        print("***", i, '/', len(imglist))
        pil_img = Image.open(imglist[i])
        pil_img.save(outpath + os.path.basename(imglist[i])[:-4]+'.jpg')

def generate_mask(imglist, outpath):
    newW, newH = 1440,900
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    img = np.zeros((900, 1440, 3), np.uint8)
    img = Image.fromarray(img)
    for i in range(len(imglist)):
        print("***", i, '/', len(imglist))
        img.save(outpath + os.path.basename(imglist[i])+'.png')


def rename(path):
    for file in os.listdir(path):
        os.rename(os.path.join(path,file),os.path.join(path,file+'.png'))

def rename_img(path):
    for file in os.listdir(path):
        os.rename(os.path.join(path,file),os.path.join(path,file[14:]))
def rename_mask(path):
    for file in os.listdir(path):
        os.rename(os.path.join(path,file),os.path.join(path,file[22:]+'.png'))
# rename_img('../../data/train/test/imgs/output/imgs')
# rename_mask('../../data/train/test/imgs/output/masks')

rename('../../data/train/test/masks/')
# imglist  = glob.glob('../../data/train/resized_BG/imgs/*.jpg')
# imglist.sort()
# outpath = '../../data/train/resized_BG/masks/'


# imglist  = glob.glob('../../data/train/orign/masks/*')
# imglist.sort()
# outpath = '../../data/train/orign/masks_dilation_resized_900_1440/'

# resize(imglist, outpath)
# resize_mask(imglist, outpath)
# dilation(glob.glob(outpath+'*'),outpath)
# dilation(imglist,outpath)