from scipy import io as sio


def loadmat(path):
    data_dict = sio.loadmat(path)
    return data_dict[list(data_dict.keys())[-1]]
