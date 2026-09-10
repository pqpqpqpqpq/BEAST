import sys
import os
import argparse
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import torch
import numpy as np
import torch.optim as optim
import time
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
import gc

from model.ST_GCN_AltFormer import ST_GCN_AltFormer
from dataset.utils import kmer_parser,cv_folds
from dataset import kmer_chemistry
from scipy.stats import pearsonr
from sklearn.model_selection import ShuffleSplit, train_test_split

# 全局设备对象，在 __main__ 中根据 --device 初始化
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 全局 k-mer 长度与核酸类型，在 __main__ 中根据 --kmer-len / --n-type 初始化
KMER_LEN = 6
N_TYPE = 'DNA'


def init():

    model = init_model()
    model_solver = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    criterion = torch.nn.MSELoss()
    return model, model_solver, criterion

def init_model():

    num_joints = 22 if str(N_TYPE).upper() == 'DNA' else 23
    model = ST_GCN_AltFormer(channel=8, backbone_in_c=128, num_frame=KMER_LEN,
                             num_joints=num_joints, style='ST')
    if DEVICE.type == 'cuda':
        model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)

    return model


def model_foreward(sample_batched, model,criterion):

    data = sample_batched['X'].float().to(DEVICE)
    A_batched = sample_batched['A'].float().to(DEVICE)
    label = sample_batched['pA'].to(DEVICE)
    label = torch.autograd.Variable(label, requires_grad=False)
    label = label.unsqueeze(1)

    score,_,_ = model(data,A_batched)
    score = score.to(dtype=torch.float64)
    loss = criterion(score, label)
    Rmse,r = get_acc(score, label)
    return score, loss, Rmse,r

def model_predict(X,A,pA,model,criterion):
    model.eval()
    with torch.no_grad():
        X = X.to(DEVICE)
        A = A.to(DEVICE)
        label = torch.tensor(pA).float().to(DEVICE)
        label = torch.autograd.Variable(label, requires_grad=False)
        label = label.unsqueeze(1)

        score,_,_ = model(X, A)
        score = score.to(dtype=torch.float64)
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

def fold_training(model,criterion,train_loader,val_loader,train_split,fold_index,key):
    min_rmse = float('inf')
    max_r = 0
    no_improve_epoch = 0
    n_iter = 0
    best_epoch = 0
    train_losses = []
    test_losses = []
    weight_dir = os.path.join(model_fold, key)
    for epoch in range(400):
        print("\ndna_mod_pred training.............")
        model.train()
        start_time = time.time()
        train_rmse = 0
        train_r = 0
        train_loss = 0
        for i, sample_batched in enumerate(train_loader):

            score, loss, rmse ,r = model_foreward(sample_batched, model, criterion)

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


        print("*** SHREC  Epoch: [%2d] time: %4.4f, "
              "cls_loss: %.4f  train_RMSE: %.6f ***  train_r: %.6f ***"
              % (epoch + 1, time.time() - start_time,
                 train_loss.data, train_rmse,train_r))
        start_time = time.time()

        # ***********evaluation***********
        with torch.no_grad():
            val_loss = 0
            acc_sum = 0
            model.eval()
            for i, sample_batched in enumerate(val_loader):
                label = sample_batched["pA"]
                score, loss, rmse ,r = model_foreward(sample_batched, model, criterion)
                val_loss += loss

                if i == 0:
                    score_list = score
                    label_list = label
                else:
                    score_list = torch.cat((score_list, score), 0)
                    label_list = torch.cat((label_list, label), 0)

                del score, loss, rmse, r
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()

            test_loss = val_loss / float(i + 1)
            test_rmse,test_r = get_acc(score_list, label_list)

            test_losses.append(test_loss)

            print("*** SHREC  Epoch: [%2d], "
                  "val_loss: %.6f,"
                  "val_RMSE: %.6f ***"
                  "val_r: %.6f ***"
                  % (epoch + 1, test_loss, test_rmse,test_r))

        # save best model
        if test_rmse < min_rmse:
            min_rmse = test_rmse
            max_r = test_r
            no_improve_epoch = 0
            test_rmse = round(test_rmse, 10)
            os.makedirs(weight_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(weight_dir, f'fold_{fold_index}_best.pth'))
            print("performance improve in train dataset, saved the new model......best rmse: {}".format(min_rmse))
            best_epoch = epoch + 1
        else:
            no_improve_epoch += 1
            print("no_improve_epoch: {} best rmse {} best r {}".format(no_improve_epoch, min_rmse,max_r))

        if no_improve_epoch > 15:
            print("stop training....")
            break

        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    model_path = os.path.join(weight_dir, f'fold_{fold_index}_best.pth')
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print('load best model success')

    return model,train_losses,test_losses




torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='BEAST mixed k-mer model training (Canonical + Modified)')

    parser.add_argument('--fn', type=str, default='./kmer_models/Canonical.model',
                        help='Path to the Canonical k-mer model file (default: ./kmer_models/Canonical.model)')
    parser.add_argument('--fn_M', type=str, default='./kmer_models/5mC_OnlyM.model',
                        help='Path to the Modified k-mer model file (default: ./kmer_models/5mC_OnlyM.model)')
    parser.add_argument('--model_fold', type=str, default='../train_mixed_kmer',
                        help='Directory to save model weights (default: ../train_mixed_kmer)')
    parser.add_argument('--result_fold', type=str, default='../train_mixed_kmer/result',
                        help='Directory to save CV results (default: ../train_mixed_kmer/result)')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device index (e.g. 0, 1) or "cpu" (default: 0)')
    parser.add_argument('--kmer-len', type=int, default=6,
                        help='k-mer length used to build the model (default: 6)')
    parser.add_argument('--n-type', type=str, default='DNA', choices=['DNA', 'RNA'],
                        help='Nucleotide type: DNA or RNA (default: DNA)')

    args = parser.parse_args()


    # ---------- device setup ----------
    if args.device.lower() == 'cpu':
        DEVICE = torch.device('cpu')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    KMER_LEN = args.kmer_len
    N_TYPE = args.n_type

    print(f"\ndna_mod_pred...... (device: {DEVICE}, kmer_len: {KMER_LEN}, n_type: {N_TYPE})")

    # .........inital
    print("\ninit.............")
    #........inital data and training
    # ........inital data and training
    model_fold = args.model_fold
    local_out = args.result_fold

    os.makedirs(model_fold, exist_ok=True)
    os.makedirs(local_out, exist_ok=True)
    out = 'model_weight.npy'
    fn = args.fn
    fn_M = args.fn_M
    tag = os.path.splitext(os.path.basename(fn_M))[0].split("_")[0]
    kmer_list, pA_train, labels = kmer_parser(fn)
    all_bases = ''.join(list(kmer_list))
    kmer_M_list, pA_test, _ = kmer_parser(fn_M)

    res_dict = {}

    train_splits = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    for train_split in train_splits:
        key = str(round(train_split, 2))
        print('running %s 50-fold...' % key, flush=True)
        splitter = ShuffleSplit(n_splits=5, train_size=train_split, random_state=42).split(kmer_M_list)

        res_dict[key] = {
                         'r_train': [],
                         'r_test': [],
                         'rmse_train': [],
                         'rmse_test': [],
                         }

        for fold_idx, (train_idx, test_idx) in enumerate(splitter):

            train_dna_mod_kmers = kmer_M_list[train_idx]
            train_dna_mod_pA_list = pA_test[train_idx]

            test_dna_mod_kmers = kmer_M_list[test_idx]
            test_dna_mod_pA_list = pA_test[test_idx]

            # Split 10% from modified training data as validation set (no canonical)
            train_mod_kmers, val_mod_kmers, train_mod_pA, val_mod_pA = train_test_split(
                train_dna_mod_kmers, train_dna_mod_pA_list, test_size=0.1, random_state=42)

            # Training set = all canonical + 90% modified training data
            train_kmers = np.concatenate([kmer_list, train_mod_kmers], axis=0)
            pA_list_train = np.concatenate([pA_train, train_mod_pA], axis=0)

            # Validation set = 10% modified training data only
            val_kmers = val_mod_kmers
            pA_list_val = val_mod_pA

            # Test set = remaining modified data (unchanged)
            test_kmers = test_dna_mod_kmers
            pA_list_test = test_dna_mod_pA_list

            # Save dataset splits
            dataset_dir = os.path.join(model_fold, 'dataset', key)
            os.makedirs(dataset_dir, exist_ok=True)
            np.save(os.path.join(dataset_dir, f'fold_{fold_idx}_train_kmers.npy'), train_kmers)
            np.save(os.path.join(dataset_dir, f'fold_{fold_idx}_val_kmers.npy'), val_kmers)
            np.save(os.path.join(dataset_dir, f'fold_{fold_idx}_test_kmers.npy'), test_kmers)

            A_train, X_train = kmer_chemistry.get_AX(train_kmers, n_type=N_TYPE)
            A_val, X_val = kmer_chemistry.get_AX(val_kmers, n_type=N_TYPE)
            A_test, X_test = kmer_chemistry.get_AX(test_kmers, n_type=N_TYPE)

            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            A_train = torch.tensor(A_train, dtype=torch.float32)
            A_val = torch.tensor(A_val, dtype=torch.float32)
            A_test = torch.tensor(A_test, dtype=torch.float32)

            kmer_train_data = {}
            kmer_val_data = {}
            kmer_test_data = {}
            for j in range(A_train.shape[0]):
                kmer_train_data[j] = {'X': X_train[j], 'A': A_train[j], 'pA': pA_list_train[j]}
            for j in range(A_val.shape[0]):
                kmer_val_data[j] = {'X': X_val[j], 'A': A_val[j], 'pA': pA_list_val[j]}
            for j in range(A_test.shape[0]):
                kmer_test_data[j] = {'X': X_test[j], 'A': A_test[j], 'pA': pA_list_test[j]}

            train_loader = torch.utils.data.DataLoader(kmer_train_data, batch_size=32, shuffle=True,
                                                       num_workers=8, pin_memory=False)

            val_loader = torch.utils.data.DataLoader(kmer_val_data, batch_size=32, shuffle=True,
                                                      num_workers=8, pin_memory=False)

            test_loader = torch.utils.data.DataLoader(kmer_test_data, batch_size=32, shuffle=True,
                                                      num_workers=8, pin_memory=False)

            print("data down")

            model, model_solver, criterion = init()


            model, train_losses, test_losses = fold_training(model, criterion, train_loader,val_loader,train_split,fold_idx,key)

            train_score, train_loss, train_rmse, train_r = model_predict(X_train, A_train, pA_list_train, model, criterion)
            test_score, test_loss, test_rmse, test_r = model_predict(X_test, A_test, pA_list_test, model, criterion)

            res_dict[key]['r_train'] += [train_r]
            res_dict[key]['r_test'] += [test_r]

            res_dict[key]['rmse_train'] += [train_rmse]
            res_dict[key]['rmse_test'] += [test_rmse]

            print('write down')

            print(f'finished with average results:')
            print(f'Train r: {train_r:.4f}, Test r: {test_r:.4f}')
            print(f'Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}')

            fold_file = f"{os.path.join(local_out, out)}_fold_{fold_idx}_train_split_{train_split}.npy"
            np.save(fold_file, res_dict)
            print(f"Fold {fold_idx} saved to {fold_file}")

            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

        fold_file = f"{os.path.join(local_out, out)}{tag}- {train_split}.npy"
        np.save(fold_file, res_dict)
        print(f"{tag}- {train_split} saved to {fold_file}")
