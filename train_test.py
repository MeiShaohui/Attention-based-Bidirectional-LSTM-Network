import os
import torch
import time
import numpy as np
import deepdish as dd
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import cohen_kappa_score
from preprocess import mk_joint_ten_fold_dataset
from dataset import IndexDataset
from network import SpatialAttention, SpectralAttention, BiLSTM


def mk_train_test_index_map(index_map, t):
    index_map_train = {}
    index_map_test = {}
    for k in index_map:
        index_map_test[k] = []
        for i in range(10):
            if i != t:
                index_map_test[k].extend(index_map[k][i])
            else:
                index_map_train[k] = index_map[k][i]

    return index_map_train, index_map_test


def main():
    CHECK_POINT_DIR = os.path.join("check_point", "ours-joint",
                                   datetime.now().strftime("%Y-%m-%d-%H-%M"))
    DATASETS_MOST = {
        "Pavia": 9,
        "PaviaU": 9,
        "Salinas": 16,
        "Indian_Pines": 16,
    }
    PATCH_SIZE = 9
    TRAIN_TIME = 500
    EPOCH_TIME = 10
    LR = 1e-3
    LR_WEIGHT_DECAY = 5e-4
    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 128

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result_dict = mk_joint_ten_fold_dataset(datasets=DATASETS_MOST,
                                            patch_size=PATCH_SIZE)
    for DATASET in DATASETS_MOST:
        padded_data = result_dict[DATASET]["padded_data"].astype(np.float32)
        index_map = result_dict[DATASET]["ten_fold_index_map"]
        input_c = padded_data.shape[2]
        class_n = DATASETS_MOST[DATASET]
        aa_list = []
        oa_list = []
        kappa_list = []
        acc_list = []
        train_time_list = []
        test_time_list = []

        for t in range(TRAIN_TIME):
            index_map_train, index_map_test = mk_train_test_index_map(
                index_map, t)
            train_dataset = IndexDataset(padded_data,
                                         index_map_train,
                                         PATCH_SIZE,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize(0.0, 1.0)
                                         ]))
            test_dataset = IndexDataset(padded_data,
                                        index_map_test,
                                        PATCH_SIZE,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.0, 1.0)
                                        ]))
            del index_map_train, index_map_test
            model = nn.Sequential(
                SpatialAttention(input_c, PATCH_SIZE), SpectralAttention(),
                BiLSTM(hidden_size=64,
                       num_layers=2,
                       dropout=0.5,
                       n_class=class_n)).to(DEVICE)
            criterion = nn.CrossEntropyLoss()

            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=LR,
                                         weight_decay=LR_WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.1,
                patience=50,
                threshold=0.05,
                verbose=True,
                cooldown=10,
                min_lr=1e-10,
                eps=1e-15)
            best_oa = 0
            best_state = None
            e_train_time_list = []
            e_test_time_list = []
            print("Start Train with Dataset {}, Ten Folded, Time {}".format(
                DATASET, t))
            for e in range(EPOCH_TIME):
                trainset_len = len(train_dataset)
                validset_len = int(0.1 * trainset_len)
                trainset, validset = random_split(
                    train_dataset, [trainset_len - validset_len, validset_len])
                del trainset_len, validset_len

                trainset_loader = DataLoader(trainset,
                                             batch_size=TRAIN_BATCH_SIZE,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=8)
                validset_loader = DataLoader(validset,
                                             batch_size=TEST_BATCH_SIZE,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=8)
                testset_loader = DataLoader(test_dataset,
                                            batch_size=TEST_BATCH_SIZE,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=8)
                model.train()

                epoch_loss_avr = 0
                train_t0 = time.perf_counter()

                for i, (x, y) in enumerate(trainset_loader):
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)

                    out = model(x)
                    loss = criterion(out, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss_avr += loss.item()

                train_t1 = time.perf_counter()
                e_train_time_list.append(train_t1 - train_t0)
                epoch_loss_avr /= (i + 1)
                print("Epoch: {}\n\tAverage Loss: {}".format(
                    e + 1, epoch_loss_avr))
                scheduler.step(epoch_loss_avr)
                del train_t0, train_t1

                model.eval()
                e_oa_c = 0
                e_oa_t = 0
                test_t0 = time.perf_counter()

                for (x, y) in validset_loader:
                    with torch.no_grad():
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)

                        out = model(x)

                        _, predict = torch.max(out, 1)
                        e_oa_t += y.size()[0]
                        e_oa_c += (predict == y).sum()

                test_t1 = time.perf_counter()
                e_test_time_list.append(test_t1 - test_t0)
                e_oa = e_oa_c / e_oa_t
                print("\tValidation OA: {}".format(e_oa))
                if e_oa >= best_oa:
                    best_oa = e_oa
                    best_state = model.state_dict()
                    print("\tCurrent Best...")
                del test_t0, test_t1
                del predict, e_oa_c, e_oa_t, e_oa

            train_time_list.append(e_train_time_list)
            test_time_list.append(e_test_time_list)
            del e_train_time_list, e_test_time_list
            print("Start Test with Dataset {}, Ten Folded, Time {}".format(
                DATASET, t))
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                correct = {}
                total = {}
                t_oa_c = 0
                t_oa_t = 0
                y_pred = []
                y_true = []
                for (x, y) in testset_loader:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    out = model(x)

                    _, predict = torch.max(out, 1)
                    y_pred.extend(predict.tolist())
                    y_true.extend(y.tolist())
                    t_oa_t += y.size()[0]
                    t_oa_c += (predict == y).sum().item()
                    for e, p in zip(y, predict):
                        e, p = e.item(), p.item()
                        if e in total:
                            total[e] += 1
                            if e == p:
                                correct[e] += 1
                        else:
                            total[e] = 1
                            if e == p:
                                correct[e] = 1
                            else:
                                correct[e] = 0
                t_oa = t_oa_c / t_oa_t
                t_acc = {}
                for k in total:
                    t_acc[k] = correct[k] / total[k]
                t_aa = 0
                for k in t_acc:
                    t_aa += t_acc[k]
                t_aa /= len(t_acc)
                t_kappa = cohen_kappa_score(np.array(y_true), np.array(y_pred))

                print("\tSaving Test Result...")
                if not os.path.exists(CHECK_POINT_DIR):
                    os.makedirs(CHECK_POINT_DIR)
                dd.io.save(os.path.join(
                    CHECK_POINT_DIR,
                    "{}_{}_{}.h5".format(DATASET, PATCH_SIZE, t)), {
                        "oa": t_oa,
                        "aa": t_aa,
                        "kappa": t_kappa,
                        "acc": t_acc,
                    },
                           compression=("blosc", 9))
                save_path_base = "{}_{}_{}".format(DATASET, PATCH_SIZE, t)
                torch.save(
                    best_state,
                    os.path.join(CHECK_POINT_DIR, save_path_base + ".pth"))
                print("\tDone")

                oa_list.append(t_oa)
                aa_list.append(t_aa)
                kappa_list.append(t_kappa)
                acc_list.append(t_acc)

                del correct, total, t_oa_c, t_oa_t, y_pred, y_true, t_oa, t_acc, t_aa, t_kappa
        print("Saving Test Result...")
        dd.io.save(os.path.join(CHECK_POINT_DIR,
                                "{}_{}_List.h5".format(DATASET, PATCH_SIZE)), {
                                    "oa_list": oa_list,
                                    "aa_list": aa_list,
                                    "kappa_list": kappa_list,
                                    "acc_list": acc_list,
                                    "train_time_list": train_time_list,
                                    "test_time_list": test_time_list,
                                },
                   compression=("blosc", 9))
        print("Done")


if __name__ == "__main__":
    main()
