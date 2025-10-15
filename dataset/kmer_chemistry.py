import torch
from rdkit import Chem
# from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from collections import defaultdict
import numpy as np
import itertools



def get_kmer_smiles(k, base):

    bs = []
    sm = []
    for b, s in base.items():
        bs.append(b)
        sm.append(s)
    bs = list(itertools.product(*([bs] * k)))
    sm = list(itertools.product(*([sm] * k)))

    smiles = defaultdict(list)
    for i in list(range(0, len(bs))):
        smiles[''.join(bs[i])].append(sm[i])

    return smiles

def get_n_hydro(smiles):
    '''
    get number of Hs
    '''
    mol = Chem.MolFromSmiles(smiles)
    before = mol.GetNumAtoms()
    mol = Chem.AddHs(mol)
    after = mol.GetNumAtoms()
    nH = after - before
    return nH

def get_compound_graph(smiles, Atms):

    X_list = []
    A_list = []
    for sm in smiles:
        mol = Chem.MolFromSmiles(sm)

        X = np.zeros((mol.GetNumAtoms(), len(Atms) + 4))

        A = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()))

        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            symbol_idx = Atms.index(symbol)
            atom_degree = len(atom.GetBonds())
            implicit_valence = mol.GetAtomWithIdx(atom_idx).GetImplicitValence()
            X[atom_idx, symbol_idx] = 1
            X[atom_idx, len(Atms)] = atom_degree
            X[atom_idx, len(Atms) + 1] = get_n_hydro(symbol)
            X[atom_idx, len(Atms) + 2] = implicit_valence
            if mol.GetAtomWithIdx(atom_idx).GetIsAromatic():
                X[atom_idx, len(Atms) + 3] = 1

            for n in (atom.GetNeighbors()):
                neighbor_atom_idx = n.GetIdx()
                A[atom_idx, neighbor_atom_idx] = 1

        X_list.append(X)
        A_list.append(A)

    return A_list, X_list

def pad_compound_graph(mat_list, nAtms, axis=None):
    '''
    MutliGraphCNN assumes that the number of nodes for each graph in the dataset is same.
    for graph with arbitrary size, we append all-0 rows/columns in adjacency and feature matrices and based on max graph size
    function takes in a list of matrices, and pads them to the max graph size
    assumption is that all matrices in there should be symmetric (#atoms x #atoms)
    output is a concatenated version of the padded matrices from the lsit

    '''
    assert type(mat_list) is list
    padded_matrices = []
    for m in mat_list:
        for bs in m:
            bs = bs.tolist()
            pad_length = nAtms - len(bs)
            if axis == 0:
                padded_matrices += [np.pad(bs, [(0, pad_length), (0, 0)], mode='constant')]
            elif axis is None:
                padded_matrices += [np.pad(bs, (0, pad_length), mode='constant')]

    return np.vstack(padded_matrices)


def get_AX_matrix(smiles, Atms, nAtms, k):

    A_mat_list = []
    X_mat_list = []
    for sm in smiles:
        A, X = get_compound_graph(sm, Atms)
        A_mat_list += [A]
        X_mat_list += [X]

    padded_A_mat = pad_compound_graph(A_mat_list, nAtms)

    padded_X_mat = pad_compound_graph(X_mat_list, nAtms, axis=0)

    padded_A_mat = np.split(padded_A_mat, len(smiles),axis=0)


    padded_A_mat = np.array(padded_A_mat)
    padded_A_mat = padded_A_mat.reshape(len(padded_A_mat),k,22,22)

    padded_X_mat = np.split(padded_X_mat, len(smiles), axis=0)
    padded_X_mat = np.array(padded_X_mat)
    padded_X_mat = padded_X_mat.reshape(len(padded_X_mat), k, 22, 8)


    return padded_A_mat, padded_X_mat
    #padded_A_mat[num_kmer,k,23,23]
    #padded_X_mat[num_kmer,k,23,8]


def get_AX(kmer_list, n_type="DNA", return_smiles=False):

    k = len(kmer_list[0])

    dna_base = {"A": "OP(=O)(O)OCC1OC(N3C=NC2=C(N)N=CN=C23)CC1",
                "T": "OP(=O)(O)OCC1OC(N2C(=O)NC(=O)C(C)=C2)CC1",
                "G": "OP(=O)(O)OCC1OC(N2C=NC3=C2N=C(N)NC3=O)CC1",
                "C": "OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C=C2)CC1",
                "M": "OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C(C)=C2)CC1",  # 5mC
                "Q": "OP(=O)(O)OCC1OC(N3C=NC2=C(NC)N=CN=C23)CC1",  # 6mA
                'K': "OP(=O)(O)OCC1OC(N2C(=O)N=C(N)C(CO)=C2)CC1"}  # 5hmC

    if n_type == "DNA":
        smiles = get_kmer_smiles(k, dna_base)
        smiles = [smiles.get(kmer)[0] for kmer in kmer_list]

        A, X = get_AX_matrix(smiles, ['C', 'N', 'O', 'P'], 22,k)
        # A（133，133），X（133,8）

    if return_smiles:
        return A, X, smiles
    return A, X

