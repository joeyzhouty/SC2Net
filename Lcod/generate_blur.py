#python scripts
__author__='Du Jiawei'
#Descrption:
import cv2
import os.path as osp
import numpy as np
from PIL import Image
from glob import glob
Source_Folder = "/home/dujw/darse/save_exp/fruit"

def transform(Source):
    for img in glob(osp.join(Source,"*.jpeg")):
        print "preprocessing",img
        img_blur = GaussBlur(img)
        im = Image.fromarray(img_blur,"RGB")
        # im.save(osp.join(Source,"blur",img.split("/")[-1]),"JPEG")
def GaussBlur(source):
    img = cv2.imread(source)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    import pdb
    pdb.set_trace()
    img_blur = cv2.GaussianBlur(img,(9,9),4)
    return img_blur

def resizeall(Source):
    for img in glob(osp.join(Source,"*.jpeg")):
        im = resize(img)
        im = Image.fromarray(im,"RGB")
        im.save(img,"JPEG")
    """
    for img in glob(osp.join(Source,"blur","*.jpeg")):
        im = resize(img)
        im = Image.fromarray(im,"RGB")
        im.save(img,"JPEG")
    for img in glob(osp.join(Source,"test","*.jpeg")):
        im = resize(img)
        im = Image.fromarray(im,"RGB")
        im.save(img,"JPEG")
    """

def resize(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    import pdb
    pdb.set_trace()
    return img[100:132,100:132,:]

def imgshow(img1,img2):
    im = Image.fromarray(img1,"RGB")
    from matplotlib import pyplot as plt
    plt.figure(figsize=(15,8))
    plt.subplot(211)
    plt.imshow(img1)
    plt.subplot(212)
    plt.imshow(img2)
    plt.show()
if __name__ == '__main__':
    resizeall(Source_Folder)
