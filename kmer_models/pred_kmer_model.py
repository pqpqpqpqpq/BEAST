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

# 全局设备对象，在 main() 中根据 --device 初始化
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def model_predict(X, A, model):
    model.eval()
    with torch.no_grad():
        X = X.to(DEVICE)
        A = A.to(DEVICE)
        score, _, _ = model(X, A)
        score = score.to(dtype=torch.float64)
    return score


def init_model(kmer_len=6, n_type='DNA'):
    num_joints = 22 if str(n_type).upper() == 'DNA' else 23
    model = ST_GCN_AltFormer(channel=8, backbone_in_c=128, num_frame=kmer_len,
                             num_joints=num_joints, style='ST')
    if DEVICE.type == 'cuda':
        model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)
    return model


def load_checkpoint(model, path):
    """Load a checkpoint, tolerating a 'module.' DataParallel prefix mismatch."""
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    keys = list(state.keys())
    has_prefix = bool(keys) and keys[0].startswith('module.')
    is_parallel = isinstance(model, torch.nn.DataParallel)
    if has_prefix and not is_parallel:
        state = {k[len('module.'):]: v for k, v in state.items()}
    elif (not has_prefix) and is_parallel:
        state = {'module.' + k: v for k, v in state.items()}
    model.load_state_dict(state)
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

    n_cols = len(data_lines[0].split())
    header_tokens = data_lines[0].split()
    expected_columns = {"kmer", "level_mean", "level_stdv", "sd_mean", "sd_stdv", "weight"}

    if set(header_tokens).intersection(expected_columns):
        print("The header row is detected and automatically skipped.")
        data_lines = data_lines[1:]

    kmer_data = [line.split() for line in data_lines]
    means = new_level_means.cpu().numpy().flatten()

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'w') as f_out:
        for line in comment_lines:
            f_out.write(line + '\n')
        if n_cols == 2:
            # two-column templates (e.g. RNA 5-mer/9-mer tables): kmer + predicted mean
            for row, v in zip(kmer_data, means):
                f_out.write("{}\t{:.8f}\n".format(row[0], v))
        else:
            df = pd.DataFrame(kmer_data, columns=["kmer", "level_mean", "level_stdv",
                                                  "sd_mean", "sd_stdv", "weight"])
            df['level_mean'] = means
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

    parser.add_argument('--device', type=str, default='0',
                        help='GPU device index (e.g. 0, 1) or "cpu" (default: 0).')

    parser.add_argument('--kmer-len', type=int, default=6,
                        help='k-mer length used to build the model (default: 6).')
    parser.add_argument('--n-type', type=str, default='DNA', choices=['DNA', 'RNA'],
                        help='Nucleotide type: DNA or RNA (default: DNA).')

    args = parser.parse_args()

    # ---------- device setup ----------
    global DEVICE
    if str(args.device).lower() == 'cpu':
        DEVICE = torch.device('cpu')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {DEVICE}")
    print(f"Input k-mer file: {args.fn}")
    print(f"Model checkpoint: {args.model_weight}")
    print(f"Template model file: {args.kmer_model_file}")
    print(f"Output path: {args.output_path}")

    model = init_model(args.kmer_len, args.n_type)
    model = load_checkpoint(model, args.model_weight)

    kmer_list, _, _ = kmer_parser(args.fn)
    print(f"k-mers processed: {len(kmer_list)}")
    A_train, X_train = kmer_chemistry.get_AX(kmer_list, n_type=args.n_type, return_smiles=False)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    A_train = torch.tensor(A_train, dtype=torch.float32)

    pA = model_predict(X_train, A_train, model)

    replace_level_mean(args.kmer_model_file, pA, args.output_path)

if __name__ == "__main__":
    main()
