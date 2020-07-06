import torch
#from sklearn.metrics import jaccard_similarity_score as jsc
from sklearn.metrics import jaccard_score as js
'''
labels = torch.tensor([[[0, 0, 2], [1, 1, 1], [1, 2, 3]],
                       [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                       [[1, 2, 3], [3, 2, 1], [0, 0, 0]]])

target = torch.tensor([[[0, 0, 2], [1, 1, 1], [1, 2, 3]],
                       [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                       [[1, 2, 3], [3, 2, 1], [0, 0, 1]]])
'''
labels = torch.randint(10, (4, 3, 5, 5))
target = labels.clone()
target[target==0] = 1

np_labels = labels.cpu().numpy().reshape(-1)
np_target = target.cpu().numpy().reshape(-1)
print(np_labels, np_target)
print(js(np_target, np_labels, average='weighted'))
