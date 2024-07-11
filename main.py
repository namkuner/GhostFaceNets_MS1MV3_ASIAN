import torch
from ellzaf_ml.models import GhostFaceNetsV2
from torchsummary import summary
IMAGE_SIZE = 112
import time
#return embedding
# model = GhostFaceNetsV2(image_size=IMAGE_SIZE, width=1.3, dropout=0.2)
# img = torch.randn(5, 3, IMAGE_SIZE, IMAGE_SIZE)
# model = model.cpu()
# img = img.cpu()
# print(img)
# model(img)
#return classification
# model = GhostFaceNetsV2(image_size=IMAGE_SIZE, num_classes=10, width=1, dropout=0.)
# img = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
# x=time.time()
# size = model(img)
# y=time.time()
# print(y-x)
# print(size.shape)
from ellzaf_ml.models.ghostfacenetsv1 import  GhostFaceNetsV1
model =GhostFaceNetsV1(image_size=IMAGE_SIZE, width=1.3, dropout=0.2)
model.cpu()
for i in range(10):
    img = torch.randn(3,3,112,112)
    img.cpu()
    x= time.time()
    model(img)
    y = time.time()
    print(y-x)



# model=model.cuda()
# summary(model,(3,112,112))
# import torch.nn as nn
# def count_batchnorm_params(model):
#     total_params = 0
#     prelu_params = 0
#     for module in model.modules():
#         if isinstance(module, nn.BatchNorm2d) or isinstance(module,nn.BatchNorm1d) :
#             for param in module.parameters():
#                 total_params += param.numel()
#         if isinstance(module,nn.PReLU):
#             prelu_params +=1
#     print("prelu_params",prelu_params)
#     return total_params
#
# # Gọi hàm và in kết quả
# num_bn_params = count_batchnorm_params(model)
# print(f"Số lượng tham số của tất cả các lớp BatchNorm2d trong mô hình: {num_bn_params}")