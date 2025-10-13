import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import random
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.backends import cudnn
import matplotlib.pyplot as plt
from IEMOCAP_Dataset import collate_fn
from ConMSDMamba import ConmsdBlock
import torch.nn.functional as F
import math
import lmdb
from tqdm import tqdm
from datetime import datetime
import json

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def mixup_data(x, y, alpha=0.2):
    """Applies mixup augmentation to the batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def save_experiment_results(model_name, seed, results, model_summary, lr, optimizer, scheduler, save_dir='experiment_results'):
    os.makedirs(save_dir, exist_ok=True)

    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    experiment_record = {
        'timestamp': timestamp,
        'model_name': model_name,
        'seed': seed,
        'results': results,
        'model_summary': model_summary.split('\n'),
        'lr': lr,
        'optimizer': optimizer.__class__.__name__,
        'scheduler': scheduler.__class__.__name__
    }

    filename = os.path.join(model_dir, f"seed{seed}_{timestamp}.json")

    with open(filename, 'w') as f:
        json.dump(experiment_record, f, indent=4)

    print(f"\n实验结果已保存到: {filename}")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False


class lmdb_dataset(Dataset):
    def __init__(self, out_path, mode):
        self.env = lmdb.open(out_path + mode)
        self.txn = self.env.begin(write=False)
        self.len = self.txn.stat()['entries']

    def __getitem__(self, index):
        key_data = 'data-%05d' % index
        key_label = 'label-%05d' % index

        data = np.frombuffer(self.txn.get(key_data.encode()), dtype=np.float32)

        data = torch.FloatTensor(data.reshape(-1, 768).copy())
        label = np.frombuffer(self.txn.get(key_label.encode()), dtype=np.int64)
        label = torch.LongTensor(label.copy()).squeeze()

        return data, label

    def __len__(self):
        return int(self.len / 2)


final_wa = []
final_ua = []

for i in range(1, 6):
    seed = 3
    set_seed(seed)
    torch.cuda.empty_cache()

    out_path = r'/IEMOCAP/emotion2vec_base/'  # 修改特征路径
    trainplace = r'train' + str(i)
    validplace = r'valid' + str(i)
    train_dataset = lmdb_dataset(out_path, trainplace)
    develop_dataset = lmdb_dataset(out_path, validplace)
    trainDataset = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    developDataset = DataLoader(
        dataset=develop_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = ConmsdBlock(
        encoder_dim=768,
        feed_forward_expansion_factor=2,
        conv_expansion_factor=2,
        feed_forward_dropout_p=0.1,
        conv_dropout_p=0.1,
        conv_kernel_size=3,
        half_step_residual=True
    ).to(device)

    if i == 1:
        print("\n============= 模型架构 =============")
        model_summary = str(model)
        print(model)
        print("==================================\n")

    lr = 5e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3, T_mult=2, eta_min=1e-4 * 0.1
    )

    loss = nn.CrossEntropyLoss().to(device)

    best_wa = 0
    best_ua = 0
    num = 0

    for epoch in range(120):
        model.train()
        loss_tr = 0.0
        start_time = time.time()
        pred_all, actu_all = [], []

        for step, (datas, labels, lengths, masks) in enumerate(trainDataset, 0):
            datas = datas.to(device)
            labels = labels.view(len(labels))
            labels = labels.to(device)
            masks = masks.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()

            mixed_datas, labels_a, labels_b, lam = mixup_data(datas, labels, alpha=0.5)
            out = model(mixed_datas, masks)
            err1 = mixup_criterion(loss, out, labels_a.long(), labels_b.long(), lam)

            err1.backward()
            optimizer.step()

            pred = torch.max(out.cpu().data, 1)[1].numpy()
            actu = labels.cpu().data.numpy()
            pred_all += list(pred)
            actu_all += list(actu)
            loss_tr += err1.cpu().item()

        loss_tr = loss_tr / len(trainDataset.dataset)
        pred_all, actu_all = np.array(pred_all), np.array(actu_all)
        wa_tr = metrics.accuracy_score(actu_all, pred_all)
        ua_tr = metrics.recall_score(actu_all, pred_all, average='macro')
        end_time = time.time()

        print('TRAIN:: Epoch: ', epoch, '| Loss: %.3f' % loss_tr, '| wa: %.3f' % wa_tr, '| ua: %.3f' % ua_tr)
        print('所耗时长:', str(end_time - start_time), 's')
        scheduler.step()

        model.eval()
        loss_de = 0.0
        start_time = time.time()
        pred_all, actu_all = [], []

        for step, (datas, labels, lengths, masks) in enumerate(developDataset, 0):
            datas = datas.to(device)
            labels = labels.view(len(labels))
            labels = labels.to(device)
            masks = masks.to(device)
            lengths = lengths.to(device)

            with torch.no_grad():
                out = model(datas, masks)
            err1 = loss(out, labels.long())
            pred = torch.max(out.cpu().data, 1)[1].numpy()
            actu = labels.cpu().data.numpy()
            pred_all += list(pred)
            actu_all += list(actu)
            loss_de += err1.cpu().item()

        loss_de = loss_de / len(developDataset.dataset)
        pred_all, actu_all = np.array(pred_all, dtype=int), np.array(actu_all, dtype=int)
        wa_de = metrics.accuracy_score(actu_all, pred_all)
        ua_de = metrics.recall_score(actu_all, pred_all, average='macro')

        if (ua_de + wa_de) >= (best_ua + best_wa):
            best_ua = ua_de
            best_wa = wa_de
            num = epoch

        end_time = time.time()
        print('VALID:: Epoch: ', epoch, '| Loss: %.3f' % loss_de, '| wa: %.3f' % wa_de, '| ua: %.3f' % ua_de)
        print('所耗时长:  ', str(end_time - start_time), 's')

    final_wa.append(best_wa)
    final_ua.append(best_ua)
    print('第%d折最好结果: | wa: %.3f' % (i, best_wa), '|ua: %.3f' % best_ua)

final_wa = np.array(final_wa)
final_ua = np.array(final_ua)
final_wa1 = np.mean(final_wa)
final_ua1 = np.mean(final_ua)
print('五折交叉验证结果:  | wa: %.3f' % final_wa1, '|ua: %.3f' % final_ua1)

results = {
    'fold_results': {
        'wa': final_wa.tolist(),
        'ua': final_ua.tolist()
    },
    'final_results': {
        'mean_wa': float(final_wa1),
        'mean_ua': float(final_ua1)
    }
}

save_experiment_results("ConMSDMamba", seed, results, model_summary, lr, optimizer, scheduler)