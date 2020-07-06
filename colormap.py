# color map source:
# https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae

import numpy as np

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_color_map(num_classes):
    cmap = color_map()
    #print(cmap.shape)
    array = []
    for i in range(num_classes):
        array.append(list(cmap[i]))

    return array

NUM_CLASSES = 22
print(get_color_map(num_classes=NUM_CLASSES))


