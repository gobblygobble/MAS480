import torch

BATCH_SIZE = 4
NUM_INDEXES = 3
IMAGE_HEIGHT = 2
IMAGE_WIDTH = 2

A = torch.randn((BATCH_SIZE, NUM_INDEXES, IMAGE_HEIGHT, IMAGE_WIDTH))
what, index_A = torch.max(A, dim=1)
print(A)
print(index_A)
print("A.size(): {}, index_A.size(): {}".format(A.size(), index_A.size()))
#print(what)