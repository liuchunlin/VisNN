import numpy as np
import torch
from matplotlib import image as mpimg
from torch.utils.data import Dataset


def classify_spiral_data(num: int, noise: float) -> (np.ndarray, np.ndarray):
    t = np.linspace(0, 1, num)
    r = t * 5
    t = t * (1.75 * 2 * np.pi) + np.arange(num) % 2 * np.pi
    x = np.zeros([num, 2])
    x[:, 0] = r * np.sin(t) + np.random.uniform(-1, 1, num) * noise * 3
    x[:, 1] = r * np.cos(t) + np.random.uniform(-1, 1, num) * noise * 3
    y = np.arange(num) % 2
    return x.astype(np.float32), y.astype(np.float32)


def classify_circle_data(num: int, noise: float) -> (np.ndarray, np.ndarray):
    radius: float = 5
    pnum: int = num//2
    r = np.hstack([np.random.uniform(0, radius*0.5, pnum),
                    np.random.uniform(radius*0.7, radius, num - pnum)])
    a = np.random.uniform(0, np.pi*2, num)
    x = np.zeros((num, 2), dtype=np.float)
    x[:, 0] = r * np.sin(a) + np.random.uniform(-radius, radius, num) * noise
    x[:, 1] = r * np.cos(a) + np.random.uniform(-radius, radius, num) * noise
    y = np.hstack([np.zeros(pnum, dtype=np.int32)+1, np.zeros(num - pnum, dtype=np.int32)])
    return x.astype(np.float32), y.astype(np.float32)


def classify_image_data(filename, num: int, noise: float)-> (np.ndarray, np.ndarray):
    img = mpimg.imread(filename)
    cx = np.random.randint([img.shape[0], img.shape[1]], size=(num, 2))
    y = img[cx[:, 0], cx[:, 1]]
    if y.ndim > 1:
        y = y[:, 0]
    x = np.zeros_like(cx, dtype=np.float32)
    radius: float = 5
    x[:, 0] = cx[:, 1] / img.shape[1] * radius + np.random.uniform(-radius, radius, num) * noise * 0.5
    x[:, 1] = cx[:, 0] / img.shape[0] * radius + np.random.uniform(-radius, radius, num) * noise * 0.5
    return x.astype(np.float32), y.astype(np.float32)


class MyDataset(Dataset):
    def __init__(self, num: int, noise: float):
        x, y = classify_spiral_data(num, noise)
        #x, y = classify_circle_data(num, noise)
        #x, y = classify_image_data('data/img_data01.png', num, 0)

        shuffle = np.arange(len(x))
        np.random.shuffle(shuffle)
        x = x[shuffle]
        y = y[shuffle]

        self.x = x
        self.y = np.zeros([y.size, 2], dtype=np.float32)
        self.y[:, 0] = 1 - y
        self.y[:, 1] = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx, :]

    def get_x_range(self):
        vmin = np.min(self.x, axis=0)
        vmax = np.max(self.x, axis=0)
        return vmin, vmax

    def get_data(self, maxnum):
        return self.x[:maxnum], self.y[:maxnum]
