
import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
from matplotlib import pyplot as plt

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth_density_map')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    # print("img path:", img_path)
    # print("ground truth path:", gt_path)
    # print(" OFFICIAL :  img, target shape in image.py : ",img.size, " - ",target.shape)
    # print("target = ",target.sum())

    if train:
        crop_size = (img.size[0]/2,img.size[1]/2)
        if random.randint(0,9)<= -1:
                        
            dx = int(random.randint(0,1)*img.size[0]*1./2)
            dy = int(random.randint(0,1)*img.size[1]*1./2)
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)
                
        
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy)) # img is cropped a half, new_w = original_w/2, new_h = original_h/2
        target = target[dy:int(crop_size[1]+dy),dx:int(crop_size[0]+dx)] # target is cropped a half, similar to img.   
        
                
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    
    
    target = cv2.resize(target,(target.shape[1]//8,target.shape[0]//8),interpolation = cv2.INTER_CUBIC)*64
    
    
    # print("POST PROCESSED:  img, target shape in image.py : ",img.size,"  -  ", target.shape)

    # print("***  target = ",target.sum())
    # print(target)
    # _c = 0
    # for i in range(target.shape[0]):  
    #     for j in range(target.shape[1]):
    #         if target[i,j]!= 0:  
    #             _c += 1

    # print("_c = ",_c)

    # _tem = target.copy()
    # _tem = _tem-np.min(_tem)
    # _tem = _tem/np.max(_tem)*255
    # _tem = _tem.astype(np.uint8)
    # print("_tem = ",_tem)
    
    # _tem = cv2.applyColorMap(_tem, cv2.COLORMAP_VIRIDIS)
    
    # plt.imshow(img)
    # plt.show()    
    # plt.imshow(_tem, alpha=1.)
    # plt.show()


    return img,target