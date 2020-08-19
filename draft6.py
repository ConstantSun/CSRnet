import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
from matplotlib import pyplot as plt
  

def count_non_zero_pixel(image): 
    _count = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]): 
            if image[i,j] - 1e-11 > 1e-12 :   
                _count += 1
    return  _count

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    print("-------------------\nIMG shape: w x h = 460 x 653--------------------\n")
    print("img path:", img_path)
    print("ground_truth: ", gt_path)
    print(" OFFICIAL :  img, target shape in image.py : ",img.size, " - ",target.shape)
    print("target = ",target.sum())


    _res = []

    for i in range(target.shape[0]):
        _line = ""
        for j in range(target.shape[1]):
            # print(target[i,j], end='')
            _line += str(target[i,j]) + ' '
        # print("")
        _res.append(_line)
    
    f=open('f1.txt','w')
    s1='\n'.join(_res)
    f.write(s1)
    f.close()

    if train:
        crop_size = (img.size[0]/2,img.size[1]/2)
        if random.randint(0,9)<= -1:
            
            
            dx = int(random.randint(0,1)*img.size[0]*1./2)
            dy = int(random.randint(0,1)*img.size[1]*1./2)
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)
                
        dx, dy = 95, 230
        print(f"dx = {dx}, dy = {dy}")
        print(f"crop_size : ", crop_size)
        # crop(left, top, right, bottom)
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        print("target xxxxxx ", target.sum())

        print(" number of elements != 0 : ", count_non_zero_pixel(target))
        print(f"\n\ntarget[{dy}:{int(crop_size[1]+dy)},{dx}:{int(crop_size[0]+dx)}]")
        target = target[dy:int(crop_size[1]+dy),dx:int(crop_size[0]+dx)]
        print("target _________________", target.sum())
        
                
        if random.random()>0.8:
            print("flip: true")
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    
    # plt.imshow(img)
    # plt.show()
    print("target shape: ", target.shape)
    print("*******************\ntarget = ", target.sum() )
    target = cv2.resize(target,( int(target.shape[1]/8), int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64
    
    print("POST PROCESSED:  img, target shape in image.py : ",img.size,"  -  ", target.shape)
    print("  - target = ",target.sum())
    
    return img ,target





# img = Image.open("img_1.png") 
# plt.imshow(img)
# plt.show()
 
# left = 0
# top = 50
# right = 30
# bottom = 150
 
# # left <= x < right, top <= y < bottom
# img_res = img.crop((left, top, right, bottom)) 
 
 
# # img_res.show()    
# plt.imshow(img_res)
# plt.show()


load_data("/home/nvhuy/hangdtth/hang/ShanghaiTech/part_A/train_data/images/IMG_149.jpg") # w = 460, h = 653

# 8x3
# a = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9],
#     [10, 11, 12],
#     [13, 14, 15],
#     [16, 17, 18],
#     [19, 20, 21],
#     [22, 23, 24]
# ])

# print(a[2:6, 1:3])




