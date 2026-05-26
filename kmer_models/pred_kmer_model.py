import os
import torch
import pandas as pd
import sys
from pathlib import Path
import argparse

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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


def replace_level_mean(model_path, new_level_means, output_path):
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

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'w') as f_out:
        for line in comment_lines:
            f_out.write(line + '\n')
        df.to_csv(f_out, sep='\t', index=False, header=False)
    print(f"Successfully saved predicted model to: {output_path}")


def main():

    parser = argparse.ArgumentParser(description="BEAST Model Inference Script for k-mer level mean prediction.")

    parser.add_argument('--model-weight', type=str, required=True,
                        help='Path to the trained BEAST model weights (.pth file).')
    parser.add_argument('--kmer-model-file', type=str, required=True,
                        help='Path to the template k-mer model file.')
    parser.add_argument('--fn', type=str, required=True,
                        help='Path to the k-mer input file.')

    parser.add_argument('--output-path', type=str, default='../pred.model',
                        help='Path to save the predicted output model (default: ../pred.model).')

    args = parser.parse_args()

    model = init_model()
    model.load_state_dict(torch.load(args.model_weight))

    kmer_list, _, _ = kmer_parser(args.fn)
    A_train, X_train = kmer_chemistry.get_AX(kmer_list, n_type='DNA', return_smiles=False)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    A_train = torch.tensor(A_train, dtype=torch.float32)

    pA = model_predict(X_train, A_train, model)

    replace_level_mean(args.kmer_model_file, pA, args.output_path)

if __name__ == "__main__":
    main()