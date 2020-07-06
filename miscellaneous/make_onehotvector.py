'''
Given a 3D tensor, computes 4D tensor with
most inner vector being a one-hot vector
2D->3D conversion is from:
https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
'''

import torch
import numpy as np

INDEX_NUM = 5

x = torch.tensor([[[1, 2, 3], [0, 0, 1]],
                  [[2, 2, 2], [1, 3, 4]],
                  [[3, 2, 1], [1, 1, 0]]])
print(x)
np_x = x.cpu().numpy()
np_x_onehot = (np.arange(INDEX_NUM) == np_x[...,None]).astype(int)
x_onehot = torch.from_numpy(np_x_onehot).float().cpu()
print(x_onehot)