import os
import random
import cv2
import numpy as np

all_img = []
neg_pairs =[]
pos_pairs =[]
all_pairs =[]
path="E:\AI\VILFWCut"
for i in os.listdir(path):

    pos=[]
    shape = None
    for j in os.listdir(os.path.join(path,i)) :

        if len(os.listdir(os.path.join(path, i))) == 2 :
            shape = cv2.imdecode(np.fromfile(os.path.join(path, i, j), dtype=np.uint8), cv2.IMREAD_UNCHANGED).shape
            if shape == (112, 112, 3):
                pos.append(os.path.join(i, j))

            if len(pos)==2 :
                pos.append(1)
                pos_pairs.append(pos)
                all_pairs.append(pos)
        all_img.append(os.path.join(i,j))
num_pairs = len(pos_pairs)
for _ in range(num_pairs):
    img1, img2 = random.sample(all_img, 2)

    while os.path.dirname(img1) == os.path.dirname(img2) or [img1,img2,0] in neg_pairs or cv2.imdecode(np.fromfile(os.path.join(path,img1 ), dtype=np.uint8), cv2.IMREAD_UNCHANGED).shape !=(112,112,3) or cv2.imdecode(np.fromfile(os.path.join(path,img2 ), dtype=np.uint8), cv2.IMREAD_UNCHANGED).shape !=(112,112,3) :
        shape1 =cv2.imdecode(np.fromfile(os.path.join(path,img2 ), dtype=np.uint8), cv2.IMREAD_UNCHANGED).shape
        shape2 =cv2.imdecode(np.fromfile(os.path.join(path,img2 ), dtype=np.uint8), cv2.IMREAD_UNCHANGED).shape
        print(shape1,shape1)
        img1, img2 = random.sample(all_img, 2)

    neg_pairs.append([img1, img2,0])
    all_pairs.append([img1,img2,0])
for i in all_pairs:
    print(i)

import pandas as pd
df = pd.DataFrame(all_pairs, columns=['img1', 'img2', 'label'])

# Lưu DataFrame ra tệp CSV
csv_file = 'output.csv'
df.to_csv(csv_file, index=False)

print(f"Dữ liệu đã được lưu vào tệp {csv_file}")
