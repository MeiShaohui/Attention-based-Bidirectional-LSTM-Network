import os
import math
import numpy as np
import deepdish as dd
from collections import Counter
from random import shuffle
from utils import loadmat


def mk_joint_ten_fold_dataset(save_dir="joint_dataset",
                              load_dir="dataset",
                              datasets={"Indian_Pines": 16},
                              patch_size: int = 9):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_dict = {}
    for dataset in datasets:
        raw_dataset_path = os.path.join(
            save_dir, "{}_{}_Raw.h5".format(dataset, patch_size))
        ten_fold_dataset_path = os.path.join(
            save_dir, "{}_{}_TenFold.h5".format(dataset, patch_size))
        dataset_result_dict = {}
        raw_dataset_dict = {}
        if not os.path.exists(raw_dataset_path):
            data = loadmat(os.path.join(load_dir, "{}.mat".format(dataset)))
            gt = loadmat(os.path.join(load_dir, "{}_gt.mat").format(dataset))
            index_map, label_map_dict = _mk_index_map(gt, datasets[dataset])
            padded_data = _mk_padded_data(data, patch_size)
            dataset_result_dict["padded_data"] = padded_data
            dataset_result_dict["label_map_dict"] = label_map_dict
            raw_dataset_dict["padded_data"] = padded_data
            raw_dataset_dict["index_map"] = index_map
            raw_dataset_dict["label_map_dict"] = label_map_dict
            dd.io.save(raw_dataset_path,
                       raw_dataset_dict,
                       compression=("blosc", 9))
            del index_map, label_map_dict, padded_data
        else:
            raw_dataset_dict = dd.io.load(raw_dataset_path)
            dataset_result_dict["padded_data"] = raw_dataset_dict[
                "padded_data"]
            dataset_result_dict["label_map_dict"] = raw_dataset_dict[
                "label_map_dict"]
        if not os.path.exists(ten_fold_dataset_path):
            index_map = raw_dataset_dict["index_map"]
            ten_fold_index_map = _mk_10_fold_index_map(index_map)
            dataset_result_dict["ten_fold_index_map"] = ten_fold_index_map
            dd.io.save(ten_fold_dataset_path,
                       ten_fold_index_map,
                       compression=("blosc", 9))
            del index_map, ten_fold_index_map
        else:
            ten_fold_dataset_dict = dd.io.load(ten_fold_dataset_path)
            dataset_result_dict["ten_fold_index_map"] = ten_fold_dataset_dict
        result_dict[dataset] = dataset_result_dict

    return result_dict


def mk_joint_dataset(save_dir="joint_dataset",
                     load_dir="dataset",
                     datasets={"Indian_Pines": 16},
                     patch_size: int = 9,
                     sample=0.1):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_dict = {}
    for dataset in datasets:
        raw_dataset_path = os.path.join(
            save_dir, "{}_{}_Raw.h5".format(dataset, patch_size))
        train_test_dataset_path = os.path.join(
            save_dir, "{}_{}_{}_TrainTest.h5".format(dataset, patch_size,
                                                     sample))
        dataset_result_dict = {}
        raw_dataset_dict = {}
        if not os.path.exists(raw_dataset_path):
            data = loadmat(os.path.join(load_dir, "{}.mat".format(dataset)))
            gt = loadmat(os.path.join(load_dir, "{}_gt.mat").format(dataset))
            index_map, label_map_dict = _mk_index_map(gt, datasets[dataset])
            padded_data = _mk_padded_data(data, patch_size)
            dataset_result_dict["padded_data"] = padded_data
            dataset_result_dict["label_map_dict"] = label_map_dict
            raw_dataset_dict["padded_data"] = padded_data
            raw_dataset_dict["index_map"] = index_map
            raw_dataset_dict["label_map_dict"] = label_map_dict
            dd.io.save(raw_dataset_path,
                       raw_dataset_dict,
                       compression=("blosc", 9))
            del index_map, label_map_dict, padded_data
        else:
            raw_dataset_dict = dd.io.load(raw_dataset_path)
            dataset_result_dict["padded_data"] = raw_dataset_dict[
                "padded_data"]
            dataset_result_dict["label_map_dict"] = raw_dataset_dict[
                "label_map_dict"]
        if not os.path.exists(train_test_dataset_path):
            index_map = raw_dataset_dict["index_map"]
            index_map_train, index_map_test = _split_index_map(
                index_map, sample)
            dataset_result_dict["index_map_train"] = index_map_train
            dataset_result_dict["index_map_test"] = index_map_test
            dd.io.save(train_test_dataset_path, {
                "index_map_train": index_map_train,
                "index_map_test": index_map_test,
            },
                       compression=("blosc", 9))
            del index_map, index_map_train, index_map_test
        else:
            train_test_dataset_dict = dd.io.load(train_test_dataset_path)
            dataset_result_dict["index_map_train"] = train_test_dataset_dict[
                "index_map_train"]
            dataset_result_dict["index_map_test"] = train_test_dataset_dict[
                "index_map_test"]
        result_dict[dataset] = dataset_result_dict

    return result_dict


def mk_disjoint_dataset(save_dir="disjoint_dataset",
                        load_dir="dataset",
                        datasets={"Indian_Pines": 16},
                        patch_size: int = 9,
                        sample=0.1,
                        sample_in_whole=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_dict = {}
    for dataset in datasets:
        raw_dataset_path = os.path.join(
            save_dir, "{}_{}_Raw.h5".format(dataset, patch_size))
        train_test_dataset_path = os.path.join(
            save_dir,
            "{}_{}_{}_{}_TrainTest.h5".format(dataset, patch_size, sample,
                                              sample_in_whole))
        dataset_result_dict = {}
        raw_dataset_dict = {}
        if not os.path.exists(raw_dataset_path):
            data = loadmat(os.path.join(load_dir, "{}.mat".format(dataset)))
            gt = loadmat(os.path.join(load_dir, "{}_gt.mat").format(dataset))
            data_disjoint_0 = np.zeros_like(data)
            data_disjoint_1 = np.zeros_like(data)
            data_disjoint_0[gt == 0] = data[gt == 0]
            data_disjoint_1[gt == 0] = data[gt == 0]
            index_map, label_map_dict = _mk_index_map(gt, datasets[dataset])
            index_map_disjoint_0, index_map_disjoint_1, data_disjoint_0, data_disjoint_1 = _mk_disjoint(
                index_map, data, data_disjoint_0, data_disjoint_1)
            padded_data_disjoint_0 = _mk_padded_data(data_disjoint_0,
                                                     patch_size)
            padded_data_disjoint_1 = _mk_padded_data(data_disjoint_1,
                                                     patch_size)
            dataset_result_dict["padded_data"] = {
                "disjoint_0": padded_data_disjoint_0,
                "disjoint_1": padded_data_disjoint_1,
            }
            dataset_result_dict["label_map_dict"] = label_map_dict
            raw_dataset_dict["padded_data"] = {
                "disjoint_0": padded_data_disjoint_0,
                "disjoint_1": padded_data_disjoint_1,
            }
            raw_dataset_dict["index_map"] = {
                "disjoint_0": index_map_disjoint_0,
                "disjoint_1": index_map_disjoint_1,
            }
            raw_dataset_dict["label_map_dict"] = label_map_dict
            dd.io.save(raw_dataset_path,
                       raw_dataset_dict,
                       compression=("blosc", 9))
            del data_disjoint_0, data_disjoint_1, index_map, label_map_dict
            del index_map_disjoint_0, index_map_disjoint_1, padded_data_disjoint_0, padded_data_disjoint_1
        else:
            raw_dataset_dict = dd.io.load(raw_dataset_path)
            dataset_result_dict["padded_data"] = raw_dataset_dict[
                "padded_data"]
            dataset_result_dict["label_map_dict"] = raw_dataset_dict[
                "label_map_dict"]
        if not os.path.exists(train_test_dataset_path):
            index_map_disjoint_0 = raw_dataset_dict["index_map"]["disjoint_0"]
            index_map_test = raw_dataset_dict["index_map"]["disjoint_1"]
            if sample_in_whole and type(sample) is float:
                index_map_train = {}
                for k in index_map_disjoint_0:
                    ia = index_map_disjoint_0[k]
                    shuffle(ia)
                    sample_len = int(sample *
                                     (len(ia) + len(index_map_test[k])))
                    ia_train = ia[:sample_len]
                    index_map_train[k] = ia_train
                del k, ia, ia_train, sample_len
            else:
                index_map_train = _split_index_map(index_map_disjoint_0,
                                                   sample)
            dataset_result_dict["index_map_train"] = index_map_train
            dataset_result_dict["index_map_test"] = index_map_test
            dd.io.save(train_test_dataset_path, {
                "index_map_train": index_map_train,
                "index_map_test": index_map_test,
            },
                       compression=("blosc", 9))
            del index_map_disjoint_0, index_map_train, index_map_test
        else:
            train_test_dataset_dict = dd.io.load(train_test_dataset_path)
            dataset_result_dict["index_map_train"] = train_test_dataset_dict[
                "index_map_train"]
            dataset_result_dict["index_map_test"] = train_test_dataset_dict[
                "index_map_test"]
        result_dict[dataset] = dataset_result_dict

    return result_dict


def _mk_padded_data(data, patch_size):
    dx = patch_size // 2
    if dx != 0:
        padded_data = np.zeros(
            (data.shape[0] + 2 * dx, data.shape[1] + 2 * dx, data.shape[2]))
        padded_data[dx:-dx, dx:-dx, :] = data
        for i in range(dx):
            padded_data[:, i, :] = padded_data[:, 2 * dx - i, :]
            padded_data[i, :, :] = padded_data[2 * dx - i, :, :]
            padded_data[:, -i - 1, :] = padded_data[:, -(2 * dx - i), :]
            padded_data[-i - 1:, :] = padded_data[-(2 * dx - i), :, :]
    else:
        padded_data = data
    return padded_data


def _mk_index_map(gt, most):
    gt_no_bg = gt[gt != 0]
    cnt = Counter(gt_no_bg).most_common(most)
    selected = [e[0] for e in cnt]
    label_map_dict = {}
    label_map_dict_rev = {}
    for i, c in enumerate(selected):
        label_map_dict[i] = c
        label_map_dict_rev[c] = i
    index_map = {}
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i, j] in selected:
                label = label_map_dict_rev[gt[i, j]]
                if label not in index_map:
                    index_map[label] = [(i, j)]
                else:
                    index_map[label].append((i, j))
    return index_map, label_map_dict


def _mk_10_fold_index_map(index_map):
    ten_fold_index_map = {}
    for k in index_map:
        tf_cl = []
        cl = index_map[k]
        cl_sp = int(math.floor(0.1 * len(cl)))
        assert cl_sp > 0
        shuffle(cl)
        for i in range(9):
            si = i * cl_sp
            tf_cl.append(cl[si:si + cl_sp])
        tf_cl.append(cl[si + cl_sp:])
        ten_fold_index_map[k] = tf_cl

    return ten_fold_index_map


def _split_index_map(index_map, sample):
    index_map_train = {}
    index_map_test = {}
    for k in index_map:
        ia = index_map[k]
        shuffle(ia)
        if type(sample) is int:
            ia_train = ia[:sample]
            ia_test = ia[sample:]
        elif type(sample) is float:
            sample_len = int(math.ceil(len(ia) * sample))
            ia_train = ia[:sample_len]
            ia_test = ia[sample_len:]
        else:
            raise RuntimeError(
                "'sample''s type should only be 'int' or 'float'!")
        index_map_train[k] = ia_train
        index_map_test[k] = ia_test

    return index_map_train, index_map_test


def _mk_disjoint(index_map, data, data_disjoint_0, data_disjoint_1):
    index_map_disjoint_0 = {}
    index_map_disjoint_1 = {}
    for label in index_map:
        index_array = index_map[label]
        array_half_size = len(index_array) // 2
        array_col_dict = {}
        for e in index_array:
            _, col = e
            if col in array_col_dict:
                array_col_dict[col].append(e)
            else:
                array_col_dict[col] = [e]
        keys = list(array_col_dict.keys())
        keys.sort()
        array_cols_disjoint_0 = []
        array_disjoint_0_size = 0
        for k in keys:
            array_cols_disjoint_0.append(k)
            array_disjoint_0_size += len(array_col_dict[k])
            if array_disjoint_0_size >= array_half_size:
                break
        array_cols_disjoint_1 = [
            k for k in keys if k not in array_cols_disjoint_0
        ]
        array_disjoint_0 = []
        array_disjoint_1 = []
        for k in array_cols_disjoint_0:
            array_disjoint_0.extend(array_col_dict[k])
        for k in array_cols_disjoint_1:
            array_disjoint_1.extend(array_col_dict[k])
        index_map_disjoint_0[label] = array_disjoint_0
        index_map_disjoint_1[label] = array_disjoint_1
        # copy data
        for i, j in array_disjoint_0:
            data_disjoint_0[i, j] = data[i, j]
        for i, j in array_disjoint_1:
            data_disjoint_1[i, j] = data[i, j]

    return index_map_disjoint_0, index_map_disjoint_1, data_disjoint_0, data_disjoint_1
