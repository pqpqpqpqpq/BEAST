import os
import torch
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset import kmer_chemistry
from dataset.utils import kmer_parser
from model.ST_GCN_AltFormer import ST_GCN_AltFormer


def model_predict(X, A, model):
    model.eval()
    with torch.no_grad():
        score, _, _ = model(X, A)
        score = score.to(dtype=torch.float64)
    return score


def init_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = ST_GCN_AltFormer(channel=8, backbone_in_c=128, num_frame=6, num_joints=22, style='ST')
    model = torch.nn.DataParallel(model).cuda()
    return model


def replace_level_mean(model_path, new_level_means, output_path='../pred.model'):
    comment_lines = []
    data_lines = []


    with open(model_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                comment_lines.append(line.strip())
            else:
                data_lines.append(line.strip())


    header_tokens = data_lines[0].split()
    expected_columns = {"kmer", "level_mean", "level_stdv", "sd_mean", "sd_stdv", "weight"}

    if set(header_tokens).intersection(expected_columns):
        print("The header row is detected and automatically skipped.")
        data_lines = data_lines[1:]


    kmer_data = [line.split() for line in data_lines]
    df = pd.DataFrame(kmer_data, columns=["kmer", "level_mean", "level_stdv", "sd_mean", "sd_stdv", "weight"])


    df['level_mean'] = new_level_means.cpu().numpy().flatten()


    with open(output_path, 'w') as f_out:
        for line in comment_lines:
            f_out.write(line + '\n')
        df.to_csv(f_out, sep='\t', index=False, header=False)


def main():

    model_weight_path = '../10% model weight/Canonical/Canonical_BEAST.pth'
    kmer_model_file = '../kmer_models/r9.4_450bps.nucleotide.6mer.template.model'   # Read the template
    fn = '../kmer_models/Canonical.model'   # Read kmer


    model = init_model()
    model.load_state_dict(torch.load(model_weight_path))


    kmer_list, _, _ = kmer_parser(fn)
    A_train, X_train = kmer_chemistry.get_AX(kmer_list, n_type='DNA', return_smiles=False)


    X_train = torch.tensor(X_train, dtype=torch.float32)
    A_train = torch.tensor(A_train, dtype=torch.float32)


    pA = model_predict(X_train, A_train, model)


    replace_level_mean(kmer_model_file, pA)


if __name__ == "__main__":
    main()
