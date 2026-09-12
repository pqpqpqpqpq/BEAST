# -*- coding: utf-8 -*-
"""BEAST leave-modification-out zero-shot / few-shot training (Supplementary Figs. S13-S15).

Modes (--mode):
  a : zero-shot 5mC   - train on canonical + all 5hmC k-mers, evaluate on the held-out 5mC table
  b : zero-shot 5hmC  - train on canonical + all 5mC k-mers, evaluate on the held-out 5hmC table
  c : few-shot 5mC    - train on canonical + all 5hmC + a fraction of 5mC k-mers
  d : few-shot 5hmC   - train on canonical + all 5mC + a fraction of 5hmC k-mers

Outputs (per fold):
  weights: <model_fold>/<key>/fold_<i>_best.pth
  splits : <model_fold>/dataset/<key>/fold_<i>_{train,val,test}_kmers.npy
  metrics: <result_fold>/model_weight.npy_fold_<i>_<key>.npy
No hard-coded paths or CUDA ids (use --device).
"""
import gc
import os
import time
import argparse
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.optim as optim
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

from model.ST_GCN_AltFormer import ST_GCN_AltFormer
from dataset.utils import kmer_parser
from dataset import kmer_chemistry
from sklearn.model_selection import ShuffleSplit, train_test_split
from scipy.stats import pearsonr

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
KMER_LEN = 6
N_TYPE = 'DNA'


def init_model():
    model = ST_GCN_AltFormer(channel=8, backbone_in_c=128, num_frame=KMER_LEN,
                             num_joints=22, style='ST')
    if DEVICE.type == 'cuda':
        model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)
    return model


def init():
    model = init_model()
    model_solver = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    criterion = torch.nn.MSELoss()
    return model, model_solver, criterion


def model_foreward(sample_batched, model, criterion):
    data = sample_batched['X'].float().to(DEVICE)
    A_batched = sample_batched['A'].float().to(DEVICE)
    label = sample_batched['pA'].to(DEVICE)
    label = torch.autograd.Variable(label, requires_grad=False)
    label = label.unsqueeze(1)
    score, _, _ = model(data, A_batched)
    score = score.to(dtype=torch.float64)
    loss = criterion(score, label)
    Rmse, r = get_acc(score, label)
    return score, loss, Rmse, r


def model_predict(X, A, pA, model, criterion):
    """Batched inference (avoids a single huge tensor / GPU OOM)."""
    model.eval()
    batch_size = 32
    n = X.shape[0]
    all_scores = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            score_batch, _, _ = model(X[i:end].to(DEVICE), A[i:end].to(DEVICE))
            all_scores.append(score_batch.cpu())
        score = torch.cat(all_scores, dim=0).to(dtype=torch.float64).to(DEVICE)
        label = torch.autograd.Variable(torch.tensor(pA).float().to(DEVICE), requires_grad=False)
        label = label.unsqueeze(1)
        loss = criterion(score, label)
        Rmse, r = get_acc(score, label)
    return score, loss, Rmse, r


def get_acc(score, labels):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    if score.ndim > 1:
        score = score.squeeze(axis=1)
    if labels.ndim > 1:
        labels = labels.squeeze(axis=1)
    Rmse = np.sqrt(np.mean((score - labels) ** 2))
    if len(score) < 2:
        return Rmse, 0.0
    pearson_coefficient, p_value = pearsonr(score, labels)
    return Rmse, pearson_coefficient


def fold_training(model, criterion, train_loader, val_loader, train_split, fold_index, key):
    min_rmse = float('inf')
    max_r = 0
    no_improve_epoch = 0
    train_losses = []
    test_losses = []
    weight_dir = os.path.join(model_fold, key)
    for epoch in range(400):
        print("\nzero/few-shot training.............")
        model.train()
        start_time = time.time()
        train_rmse = train_r = train_loss = 0
        for i, sample_batched in enumerate(train_loader):
            score, loss, rmse, r = model_foreward(sample_batched, model, criterion)
            model.zero_grad()
            loss.backward()
            model_solver.step()
            train_rmse += rmse
            train_r += r
            train_loss += loss
            del score, loss, rmse, r
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        train_rmse /= float(i + 1)
        train_r /= float(i + 1)
        train_loss /= float(i + 1)
        print("*** Epoch: [%2d] time: %4.4f, cls_loss: %.4f  train_RMSE: %.6f *** train_r: %.6f ***"
              % (epoch + 1, time.time() - start_time, train_loss, train_rmse, train_r))
        start_time = time.time()

        with torch.no_grad():
            val_loss = 0
            model.eval()
            for i, sample_batched in enumerate(val_loader):
                label = sample_batched["pA"]
                score, loss, rmse, r = model_foreward(sample_batched, model, criterion)
                val_loss += loss
                if i == 0:
                    score_list, label_list = score, label
                else:
                    score_list = torch.cat((score_list, score), 0)
                    label_list = torch.cat((label_list, label), 0)
                del score, loss, rmse, r
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
            test_loss = val_loss / float(i + 1)
            test_rmse, test_r = get_acc(score_list, label_list)
            test_losses.append(test_loss)
            print("*** Epoch: [%2d], val_loss: %.6f, val_RMSE: %.6f *** val_r: %.6f ***"
                  % (epoch + 1, test_loss, test_rmse, test_r))

        if test_rmse < min_rmse:
            min_rmse = test_rmse
            max_r = test_r
            no_improve_epoch = 0
            os.makedirs(weight_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(weight_dir, f'fold_{fold_index}_best.pth'))
            print("performance improve, saved the new model......best rmse: {}".format(min_rmse))
        else:
            no_improve_epoch += 1
            print("no_improve_epoch: {} best rmse {} best r {}".format(no_improve_epoch, min_rmse, max_r))
        if no_improve_epoch > 15:
            print("stop training....")
            break
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    model_path = os.path.join(weight_dir, f'fold_{fold_index}_best.pth')
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print('load best model success')
    return model, train_losses, test_losses


def make_loader(kmers, pA):
    A, X = kmer_chemistry.get_AX(kmers, n_type=N_TYPE)
    X = torch.tensor(X, dtype=torch.float32)
    A = torch.tensor(A, dtype=torch.float32)
    data = {j: {'X': X[j], 'A': A[j], 'pA': pA[j]} for j in range(A.shape[0])}
    loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True,
                                         num_workers=8, pin_memory=False)
    return X, A, loader


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='BEAST leave-modification-out zero-shot / few-shot training.')
    parser.add_argument('--fn', type=str, default='./kmer_models/Canonical.model',
                        help='Canonical k-mer model file (default: ./kmer_models/Canonical.model)')
    parser.add_argument('--fn_M', type=str, default='./kmer_models/5mC_OnlyM.model',
                        help='5mC k-mer model file (default: ./kmer_models/5mC_OnlyM.model)')
    parser.add_argument('--fn_K', type=str, default='./kmer_models/5hmC_OnlyK.model',
                        help='5hmC k-mer model file (default: ./kmer_models/5hmC_OnlyK.model)')
    parser.add_argument('--mode', type=str, default='a', choices=['a', 'b', 'c', 'd'],
                        help='a=zero-shot 5mC, b=zero-shot 5hmC, c=few-shot 5mC, d=few-shot 5hmC')
    parser.add_argument('--model_fold', type=str, default='../train_zero_few_shot',
                        help='Directory to save weights and dataset splits (default: ../train_zero_few_shot)')
    parser.add_argument('--result_fold', type=str, default='../train_zero_few_shot/result',
                        help='Directory to save CV results (default: ../train_zero_few_shot/result)')
    parser.add_argument('--train_splits', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
                        help='Few-shot training fractions of the target modification table '
                             '(ignored in zero-shot modes a/b)')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device index (e.g. 0, 1) or "cpu" (default: 0)')
    args = parser.parse_args()

    if args.device.lower() == 'cpu':
        DEVICE = torch.device('cpu')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_fold = args.model_fold
    local_out = args.result_fold
    os.makedirs(model_fold, exist_ok=True)
    os.makedirs(local_out, exist_ok=True)
    out = 'model_weight.npy'

    kmer_list, pA_canon, _ = kmer_parser(args.fn)
    kmer_M_list, pA_M, _ = kmer_parser(args.fn_M)      # 5mC
    kmer_K_list, pA_K, _ = kmer_parser(args.fn_K)      # 5hmC

    if args.mode in ['a', 'c']:
        target_kmers, pA_target = kmer_M_list, pA_M
        other_kmers, pA_other = kmer_K_list, pA_K
        mod_name = '5mC'
    else:
        target_kmers, pA_target = kmer_K_list, pA_K
        other_kmers, pA_other = kmer_M_list, pA_M
        mod_name = '5hmC'
    is_zero_shot = args.mode in ['a', 'b']

    print(f"\nzero/few-shot (device: {DEVICE}, mode={args.mode}, target={mod_name}, "
          f"zero_shot={is_zero_shot})")

    if is_zero_shot:
        train_splits = [0.1]
    else:
        train_splits = [float(x) for x in args.train_splits.split(',') if x.strip()]

    res_dict = {}
    for train_split in train_splits:
        if is_zero_shot:
            key = 'zero_shot_' + mod_name
        else:
            key = str(round(train_split, 2))
        print(f'Running {key} ({mod_name})...', flush=True)
        res_dict[key] = {'r_train': [], 'r_test': [], 'rmse_train': [], 'rmse_test': []}

        if is_zero_shot:
            splitter = ShuffleSplit(n_splits=5, test_size=0.9,
                                    random_state=42).split(target_kmers)
        else:
            splitter = ShuffleSplit(n_splits=5, train_size=train_split,
                                    random_state=42).split(target_kmers)

        for fold_idx, (idx_a, idx_b) in enumerate(splitter):
            if is_zero_shot:
                val_kmers = target_kmers[idx_a]
                pA_list_val = pA_target[idx_a]
                test_kmers = target_kmers[idx_b]
                pA_list_test = pA_target[idx_b]
                train_kmers = np.concatenate([kmer_list, other_kmers], axis=0)
                pA_list_train = np.concatenate([pA_canon, pA_other], axis=0)
            else:
                train_pool_kmers = target_kmers[idx_a]
                train_pool_pA = pA_target[idx_a]
                test_kmers = target_kmers[idx_b]
                pA_list_test = pA_target[idx_b]
                train_mod_kmers, val_kmers, train_mod_pA, pA_list_val = train_test_split(
                    train_pool_kmers, train_pool_pA, test_size=0.1, random_state=42)
                train_kmers = np.concatenate([kmer_list, other_kmers, train_mod_kmers], axis=0)
                pA_list_train = np.concatenate([pA_canon, pA_other, train_mod_pA], axis=0)

            dataset_dir = os.path.join(model_fold, 'dataset', key)
            os.makedirs(dataset_dir, exist_ok=True)
            np.save(os.path.join(dataset_dir, f'fold_{fold_idx}_train_kmers.npy'), train_kmers)
            np.save(os.path.join(dataset_dir, f'fold_{fold_idx}_val_kmers.npy'), val_kmers)
            np.save(os.path.join(dataset_dir, f'fold_{fold_idx}_test_kmers.npy'), test_kmers)

            X_train, A_train, train_loader = make_loader(train_kmers, pA_list_train)
            X_val, A_val, val_loader = make_loader(val_kmers, pA_list_val)
            X_test, A_test, _ = make_loader(test_kmers, pA_list_test)

            model, model_solver, criterion = init()
            model, train_losses, test_losses = fold_training(model, criterion, train_loader,
                                                             val_loader, train_split, fold_idx, key)

            _, _, train_rmse, train_r = model_predict(X_train, A_train, pA_list_train, model, criterion)
            _, _, test_rmse, test_r = model_predict(X_test, A_test, pA_list_test, model, criterion)

            res_dict[key]['r_train'] += [train_r]
            res_dict[key]['r_test'] += [test_r]
            res_dict[key]['rmse_train'] += [train_rmse]
            res_dict[key]['rmse_test'] += [test_rmse]

            print(f'Fold {fold_idx}: train RMSE {train_rmse:.4f} (r {train_r:.4f}), '
                  f'test RMSE {test_rmse:.4f} (r {test_r:.4f})')

            fold_file = f"{os.path.join(local_out, out)}_fold_{fold_idx}_{key}.npy"
            np.save(fold_file, res_dict)
            print(f"Fold {fold_idx} saved to {fold_file}")

            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        np.save(f"{os.path.join(local_out, out)}_{key}.npy", res_dict)
        print(f"{key} saved")

    np.save(os.path.join(local_out, out), res_dict)
    print('save success')
