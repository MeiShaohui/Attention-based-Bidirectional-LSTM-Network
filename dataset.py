import numpy as np
from torch.utils.data import Dataset


class IndexDataset(Dataset):
    def __init__(self,
                 padded_data,
                 index_map,
                 patch_size: int,
                 transform=None):
        self.padded_data = padded_data
        self.patch_size = patch_size
        self.transform = transform
        self.indices = []
        self.label = []
        for k in index_map:
            self.indices.extend(index_map[k])
            self.label.extend([k] * len(index_map[k]))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        patch_size = self.patch_size
        y = self.label[index]
        i, j = self.indices[index]
        x = self.padded_data[i:i + patch_size, j:j + patch_size, :]
        if self.transform is not None:
            x = self.transform(x)

        return x, y


class DisjointIndexDataset(Dataset):
    def __init__(self,
                 padded_data,
                 index_map,
                 patch_size: int,
                 transform=None):
        self.padded_data = padded_data
        self.patch_size = patch_size
        self.transform = transform
        self.indices = []
        self.label = []
        for k in index_map:
            self.indices.extend(index_map[k])
            self.label.extend([k] * len(index_map[k]))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        patch_size = self.patch_size
        dx = patch_size // 2
        H, W, C = self.padded_data.shape
        x = np.empty([patch_size, patch_size, C], dtype=self.padded_data.dtype)
        y = self.label[index]
        i, j = self.indices[index]
        for m in range(patch_size):
            for n in range(patch_size):
                p = self.padded_data[i + m, j + n, :]
                if p.sum() == 0.0:
                    # need padding
                    padded = False
                    # padding inside:
                    if not padded:
                        p = self.padded_data[i + dx * 2 - m, j + n, :]
                        if p.sum() != 0.0:
                            padded = True
                    if not padded:
                        p = self.padded_data[i + m, j + dx * 2 - n, :]
                        if p.sum() != 0.0:
                            padded = True
                    # padding outside
                    if m < dx:
                        if not padded and i - m >= 0:
                            p = self.padded_data[i - m, j + n, :]
                            if p.sum() != 0.0:
                                padded = True
                        if not padded and i + patch_size * 2 - m < H:
                            p = self.padded_data[i + patch_size * 2 - m,
                                                 j + n, :]
                            if p.sum() != 0.0:
                                padded = True
                    else:
                        if not padded and i + patch_size * 2 - m < H:
                            p = self.padded_data[i + patch_size * 2 - m,
                                                 j + n, :]
                            if p.sum() != 0.0:
                                padded = True
                        if not padded and i - m >= 0:
                            p = self.padded_data[i - m, j + n, :]
                            if p.sum() != 0.0:
                                padded = True
                    if n < dx:
                        if not padded and j - n >= 0:
                            p = self.padded_data[i + m, j - n, :]
                            if p.sum() != 0.0:
                                padded = True
                        if not padded and j + patch_size * 2 - n < W:
                            p = self.padded_data[i + m,
                                                 j + patch_size * 2 - n, :]
                            if p.sum() != 0.0:
                                padded = True
                    else:
                        if not padded and j + patch_size * 2 - n < W:
                            p = self.padded_data[i + m,
                                                 j + patch_size * 2 - n, :]
                            if p.sum() != 0.0:
                                padded = True
                        if not padded and j - n >= 0:
                            p = self.padded_data[i + m, j - n, :]
                            if p.sum() != 0.0:
                                padded = True
                x[m, n, :] = p
        if self.transform is not None:
            x = self.transform(x)

        return x, y
