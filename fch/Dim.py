from torch.utils import data


class Dim(data.Dataset):
    def __init__(self, array):
        self.dim = array

    def __getitem__(self, index):
        return self.dim[index]

    def __len__(self):
        return len(self.dim)
