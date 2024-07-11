import torch
from ellzaf_ml.models import GhostFaceNetsV2
from torchsummary import summary
IMAGE_SIZE = 112
import time
#return embedding
model = GhostFaceNetsV2(image_size=IMAGE_SIZE, width=1.3, dropout=0.2)
img = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
model = model.cuda()
img = img.cuda()
model.eval()
# model(img)
#return classification
# model = GhostFaceNetsV2(image_size=IMAGE_SIZE, num_classes=10, width=1, dropout=0.)
# img = torch.randn(3, 3, IMAGE_SIZE, IMAGE_SIZE)
x=time.time()
size = model(img)
y=time.time()
print(y-x)
print(size.shape)