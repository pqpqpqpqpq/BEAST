import torch
import numpy as np
import torch.optim as optim
import time
import argparse
import gc
from numpy.distutils.fcompiler import str2bool
from model.ST_GCN_AltFormer import ST_GCN_AltFormer
from dataset.utils import kmer_parser,cv_folds
from dataset import kmer_chemistry
from scipy.stats import pearsonr
from sklearn.model_selection import ShuffleSplit


def init():

    model = init_model()
    model_solver = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    criterion = torch.nn.MSELoss()
    return model, model_solver, criterion

def init_model():

    model = ST_GCN_AltFormer(channel=8, backbone_in_c=128, num_frame=6, num_joints=22, style='ST')
    model = torch.nn.DataParallel(model).cuda()

    return model


def model_foreward(sample_batched, model,criterion):

    data = sample_batched['X'].float()
    A_batched = sample_batched['A'].float()
    label = sample_batched['pA']
    label = label.cuda()
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
        label = torch.tensor(pA).float().cuda()
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
    score = score.squeeze()
    labels = labels.squeeze()
    Rmse = np.sqrt(np.mean((score - labels) ** 2))
    pearson_coefficient, p_value = pearsonr(score, labels)
    return Rmse, pearson_coefficient

def fold_training(model,criterion,train_loader,test_loader,train_split):
    min_rmse = 50
    max_r = 0
    no_improve_epoch = 0
    n_iter = 0
    best_epoch = 0
    train_losses = []
    test_losses = []
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
            for i, sample_batched in enumerate(test_loader):
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
            torch.save(model.state_dict(),
                       '{}/epoch_{}_rmse{}_train_split{}.pth'.format(model_fold, epoch + 1,min_rmse,train_split))
            print("performance improve in train dataset, saved the new model......best rmse: {}".format(min_rmse))
            best_epoch = epoch + 1
        else:
            no_improve_epoch += 1
            print("no_improve_epoch: {} best rmse {} best r {}".format(no_improve_epoch, min_rmse,max_r))

        if no_improve_epoch > 15:
            print("stop training....")
            break

        torch.cuda.empty_cache()

    torch.cuda.empty_cache()

    model_path = '{}/epoch_{}_rmse{}_train_split{}.pth'.format(model_fold, best_epoch,min_rmse,train_split)
    model.load_state_dict(torch.load(model_path))
    print('load best model success')

    return model,train_losses,test_losses




torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    print("\ndna_mod_pred......")

    # .........inital
    print("\ninit.............")
    #........inital data and training
    model_fold = "../train_mixed_kmer"
    local_out = '/result/'
    out = 'model_weight.npy'
    fn = '../ont_models/r9.4_180mv_450bps_6mer_DNA.model'
    fn_M = '../ont_models/r9.4_450bps.cpg.m.only.6mer.template.model'
    kmer_list, pA_train, labels = kmer_parser(fn)
    all_bases = ''.join(list(kmer_list))
    kmer_M_list, pA_test, _ = kmer_parser(fn_M)

    res_dict = {}

    train_splits = [0.1,0.3,0.5,0.7,0.9]

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

        for train_idx, test_idx in splitter:

            train_dna_mod_kmers = kmer_M_list[train_idx]
            train_dna_mod_pA_list = pA_test[train_idx]

            test_dna_mod_kmers = kmer_M_list[test_idx]
            test_dna_mod_pA_list = pA_test[test_idx]

            train_kmers = np.concatenate([kmer_list, train_dna_mod_kmers], axis=0)
            pA_list_train = np.concatenate([pA_train, train_dna_mod_pA_list], axis=0)


            test_kmers = test_dna_mod_kmers
            pA_list_test = test_dna_mod_pA_list

            A_train, X_train = kmer_chemistry.get_AX(train_kmers)
            A_test, X_test = kmer_chemistry.get_AX(test_kmers)

            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            A_train = torch.tensor(A_train, dtype=torch.float32)
            A_test = torch.tensor(A_test, dtype=torch.float32)

            kmer_train_data = {}
            kmer_test_data = {}
            for j in range(A_train.shape[0]):
                kmer_train_data[j] = {'X': X_train[j], 'A': A_train[j], 'pA': pA_list_train[j]}
            for j in range(A_test.shape[0]):
                kmer_test_data[j] = {'X': X_test[j], 'A': A_test[j], 'pA': pA_list_test[j]}

            train_loader = torch.utils.data.DataLoader(kmer_train_data, batch_size=32, shuffle=True,
                                                       num_workers=8, pin_memory=False)

            test_loader = torch.utils.data.DataLoader(kmer_test_data, batch_size=32, shuffle=True,
                                                      num_workers=8, pin_memory=False)

            print("data down")

            model, model_solver, criterion = init()


            model, train_losses, test_losses = fold_training(model, criterion, train_loader,test_loader,train_split)

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

            gc.collect()
            torch.cuda.empty_cache()

        fold_file = f"{local_out + out}_5mC- {train_split}.npy"
        np.save(fold_file, res_dict)
        print(f"5mC- {train_split} saved to {fold_file}")