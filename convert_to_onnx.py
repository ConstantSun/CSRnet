import torch
import torchvision
import os 
from model import CSRNet
from student_model import CSRNet_student

net = CSRNet_student(cpr=0.5, convert_weight='2').cuda()

# model_path ='best_model/mb2-ssd-lite-Epoch-14-Loss-1.7607919469712272.pth'
# net.load(model_path)
# net.eval()
# net.to('cuda')

# model_name = (model_path.split('/')[-1]).replace('.pth', '')
# model_path = f"onnx_model/{model_name}.onnx"
# if not os.path.exists(model_path.split('/')[0]):
#     os.mkdir(model_path.split('/')[0])
# # 1,2,300,300 -> 1:image 3:shape 300: height 300 width

model_path = "student_architechture.onnx"
dummy_input = torch.randn(1, 3, 170, 255, device='cuda')
torch.onnx.export(net, dummy_input, model_path, verbose=False, input_names=['input'], output_names=['s_out', 's_kd_list', 's_resize_list'])
