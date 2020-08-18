import cv2
import glob
import numpy as np
import os

# imglist  = glob.glob('../data/masks/*.png')
# imglist.sort()
# outpath = '../data/masks_dilation/'
#
#


# imglist  = glob.glob('../data/test/masks/*.png')
# imglist.sort()
# outpath = '../data/test/masks_dilation/'
#


def del_mask():
    imglist = glob.glob('../../data/train/test/imgs/*.jpg')
    imglist.sort()
    masklist = glob.glob('../../data/train/test/masks/*.png')
    a= 0
    print(len(imglist),len(masklist))
    for i in range(len(masklist)):
        print("***",i,'/',len(masklist))
        print(masklist[i][:-4])
        # print(imglist[i])
        print('../data/imgs/'+ os.path.basename(masklist[i])[:-4])
        if '../data/imgs/'+ os.path.basename(masklist[i])[:-4] not in imglist:
            a+=1
            print(a)
            os.remove(masklist[i])
def del_img():
    imglist = glob.glob('../data/imgs/*.jpg')
    imglist.sort()
    masklist = glob.glob('../data/masks_dilation/*')
    masklist.sort()
    print(len(masklist))
    print(len(imglist))

    a = 0
    for i in range(len(imglist)):
        print("***",i,'/',len(imglist))
        # print(masklist[i])
        # print(imglist[i])
        print('../data/masks_dilation/'+ os.path.basename(imglist[i]+'.png'))
        if '../data/masks_dilation/'+ os.path.b/data/train/test/imgsasename(imglist[i])+'.png' not in masklist:
            a+=1
            print(a)
            os.remove(imglist[i])
        print(a)


# del_img()
del_mask()