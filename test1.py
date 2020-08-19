import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
'''
a = torch.load("s_out_1.pt")
print("______")
print(a.shape)
print("total pp: ", a.sum())


b = a.cpu().data.numpy()
print("___________")
print(b[0][0])

b = b[0][0]
print("___________")

print("max: ", np.max(b))
print("min: ", np.min(b))
b = np.asarray(b)
print("b shape: ", b.shape)

_c = 0.206455
for k in range(54):      
    for l in range(87):     
        if b[k][l] - _c > 0.01: 
            print("true")

'''
teacher = torch.load("t_out_0.pt")
print(teacher.shape)

# print(teacher)

h = teacher[0][0]

print("teacher: \n", h)
print("_____________")
print(h.shape)
h = h.cpu().detach().numpy()
print(h.shape)

print(h)
_min = np.min(h)
_max = np.max(h)
h = h - _min



h = h/_max*255

map = np.array(h, dtype=np.uint8)

map = cv2.applyColorMap(map, cv2.COLORMAP_JET)


plt.imshow(map)
plt.show()