import torch
from sklearn.metrics import jaccard_score

def get_iou(labels, target):
    # receives 3D tensor (NxHxW) with values as the appropriate label
    np_labels = labels.cpu().numpy().reshape(-1)
    np_target = target.cpu().numpy().reshape(-1)
    # using 'weighted' may take a longer time than others
    return jaccard_score(np_target, np_labels, average='weighted')

def get_dice():
    raise NotImplementedError
    return 0

def early_stop(use_cuda, output, target, metric, threshold):
    '''
    output dim: 4x22x256x256
    target dim: 4x256x256
    given output and target, compute the metric (iou or dice)
    and return True if it exceeds the threshold value
    '''
    if metric == "iou":
        # output: reduce to 4x256x256 before passing to iou
        _, l = torch.max(output, dim=1)
        score = get_iou(labels=l, target=target)
        print("IoU metric score: {}".format(score))
        return (score > threshold)

    elif metric == "dice":
        score = get_dice()
        print("Dice metric score: {}".format(score))
        return (score > threshold)

    print("early_stop(): Incorrect metric... returning False by default")
    return False