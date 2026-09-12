# -*- coding: utf-8 -*-
"""BEAST training with the context-held-out split (Supplementary Table S3, Fig. S17b).

Each canonical 6-mer and all of its modification-pattern variants form one *context cluster*.
Splits are made at the cluster (canonical backbone) level so that a modified test k-mer never
shares its canonical backbone with any training entry.

If --dataset_dir is given, pre-saved splits of the form
    <dataset_dir>/<train_fraction>/fold_<i>_{train,val,test}_kmers.npy
are read (same layout that this script writes by default).
Otherwise the context-held-out splits are generated on the fly and saved under
    <model_fold>/dataset/<train_fraction>/fold_<i>_{train,val,test}_kmers.npy

Only reads/writes files; no hard-coded paths or CUDA ids (use --device).
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
KMER_LEN = 6          # 6-mer DNA experiments
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
    """Batched inference (avoids building one huge tensor / GPU OOM on large tables)."""
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
        print("\ncontext-held-out training.............")
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


def make_loader(X, A, pA):
    data = {j: {'X': X[j], 'A': A[j], 'pA': pA[j]} for j in range(A.shape[0])}
    return torch.utils.data.DataLoader(data, batch_size=32, shuffle=True,
                                       num_workers=8, pin_memory=False)


def build_loader_direct(kmers, pA):
    A, X = kmer_chemistry.get_AX(kmers, n_type=N_TYPE)
    X = torch.tensor(X, dtype=torch.float32)
    A = torch.tensor(A, dtype=torch.float32)
    return X, A


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='BEAST training with the context-held-out split (canonical-backbone grouped).')
    parser.add_argument('--fn', type=str, default='./kmer_models/Canonical.model',
                        help='Canonical k-mer model file (default: ./kmer_models/Canonical.model)')
    parser.add_argument('--fn_M', type=str, default='./kmer_models/5mC_OnlyM.model',
                        help='Modified k-mer model file (default: ./kmer_models/5mC_OnlyM.model)')
    parser.add_argument('--model_fold', type=str, default='../train_context_holdout',
                        help='Directory to save weights and dataset splits (default: ../train_context_holdout)')
    parser.add_argument('--result_fold', type=str, default='../train_context_holdout/result',
                        help='Directory to save CV results (default: ../train_context_holdout/result)')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Optional pre-saved context splits <dir>/<train_fraction>/fold_<i>_*.npy; '
                             'if omitted, splits are generated and saved under <model_fold>/dataset/.')
    parser.add_argument('--train_splits', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9',
                        help='Comma-separated training fractions of the modified table (default 0.1..0.9)')
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
    kmer_M_list, pA_M, _ = kmer_parser(args.fn_M)
    mod_letter = 'M' if any('M' in k for k in kmer_M_list) else 'K'
    canon = lambda k: k.replace(mod_letter, 'C')

    pA_dict = dict(zip(kmer_list, pA_canon))
    pA_dict.update(dict(zip(kmer_M_list, pA_M)))

    # unique canonical backbones of the modified table (context clusters)
    backbones = np.unique(np.array([canon(k) for k in kmer_M_list]))
    print(f"\ncontext-held-out (device: {DEVICE}, mod: {mod_letter}, "
          f"modified kmers: {len(kmer_M_list)}, clusters: {len(backbones)})")

    train_splits = [float(x) for x in args.train_splits.split(',') if x.strip()]
    res_dict = {}

    for train_split in train_splits:
        key = str(round(train_split, 2))
        print(f'running {key} ...', flush=True)
        res_dict[key] = {'r_train': [], 'r_test': [], 'rmse_train': [], 'rmse_test': []}

        if args.dataset_dir:
            splits = None   # read pre-saved below
        else:
            splitter = ShuffleSplit(n_splits=5, train_size=train_split,
                                    random_state=42).split(backbones)
            splits = list(splitter)

        for fold_idx in range(5):
            if args.dataset_dir:
                split_dir = os.path.join(args.dataset_dir, key)
                train_kmers_all = np.load(os.path.join(split_dir, f'fold_{fold_idx}_train_kmers.npy'),
                                          allow_pickle=True)
                val_kmers = np.load(os.path.join(split_dir, f'fold_{fold_idx}_val_kmers.npy'),
                                    allow_pickle=True)
                test_kmers = np.load(os.path.join(split_dir, f'fold_{fold_idx}_test_kmers.npy'),
                                     allow_pickle=True)
                train_modified = np.array([k for k in train_kmers_all if mod_letter in k])
                train_canonical = np.array([canon(k) for k in train_modified])
                train_kmers = np.concatenate([train_canonical, train_modified])
            else:
                tr_bb, te_bb = splits[fold_idx]
                tr_backbones = set(backbones[tr_bb])
                te_backbones = set(backbones[te_bb])
                train_modified = np.array([k for k in kmer_M_list if canon(k) in tr_backbones])
                test_kmers = np.array([k for k in kmer_M_list if canon(k) in te_backbones])
                # hold out 10% of the training modified kmers for validation
                train_modified, val_kmers = train_test_split(train_modified, test_size=0.1,
                                                             random_state=42)
                train_canonical = np.array([canon(k) for k in train_modified])
                train_kmers = np.concatenate([train_canonical, train_modified])

            pA_list_train = np.array([pA_dict[k] for k in train_kmers])
            pA_list_val = np.array([pA_dict[k] for k in val_kmers])
            pA_list_test = np.array([pA_dict[k] for k in test_kmers])

            dataset_dir = os.path.join(model_fold, 'dataset', key)
            os.makedirs(dataset_dir, exist_ok=True)
            np.save(os.path.join(dataset_dir, f'fold_{fold_idx}_train_kmers.npy'), train_kmers)
            np.save(os.path.join(dataset_dir, f'fold_{fold_idx}_val_kmers.npy'), val_kmers)
            np.save(os.path.join(dataset_dir, f'fold_{fold_idx}_test_kmers.npy'), test_kmers)

            X_train, A_train = build_loader_direct(train_kmers, pA_list_train)
            X_val, A_val = build_loader_direct(val_kmers, pA_list_val)
            X_test, A_test = build_loader_direct(test_kmers, pA_list_test)

            train_loader = make_loader(X_train, A_train, pA_list_train)
            val_loader = make_loader(X_val, A_val, pA_list_val)

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

            fold_file = f"{os.path.join(local_out, out)}_fold_{fold_idx}_train_split_{train_split}.npy"
            np.save(fold_file, res_dict)
            print(f"Fold {fold_idx} saved to {fold_file}")

            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

    np.save(os.path.join(local_out, out), res_dict)
    print('save success')
