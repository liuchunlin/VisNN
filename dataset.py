import numpy as np
import torch
from torch.utils.data import Dataset


def classify_spiral_data(num: int, noise: float) -> (np.ndarray, np.ndarray):
    t = np.linspace(0, 1, num)
    r = t * 5
    t = t * (1.75 * 2 * np.pi) + np.arange(num) % 2 * np.pi
    x = np.zeros([num, 2])
    x[:, 0] = r * np.sin(t) + np.random.uniform(-1, 1, num) * noise
    x[:, 1] = r * np.cos(t) + np.random.uniform(-1, 1, num) * noise
    y = np.arange(num) % 2 * 2 - 1
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
    y = np.hstack([np.zeros(pnum, dtype=np.int32)+1, np.zeros(num - pnum, dtype=np.int32)-1])
    return x.astype(np.float32), y.astype(np.float32)


class MyDataset(Dataset):
    def __init__(self, num: int, noise: float):
        x, y = classify_spiral_data(num, noise)
        #x, y = classify_circle_data(num, noise)

        shuffle = np.arange(len(x))
        np.random.shuffle(shuffle)
        x = x[shuffle]
        y = y[shuffle]

        self.x = x
        self.y = np.zeros([y.size, 2], dtype=np.float32)
        self.y[y == -1, 0] = 1
        self.y[y != -1, 1] = 1

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
