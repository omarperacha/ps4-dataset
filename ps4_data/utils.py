import os
import pickle
import numpy as np
import pandas as pd
import random
from copy import deepcopy

import torch

TRAIN_SAMPLES = 17799
VALID_SAMPLES = 932
TEST_SAMPLES = 495
NUM_SAMPLES = TRAIN_SAMPLES + VALID_SAMPLES

SS_CLASSES = 8

PROTTRANS_EMBS_PATH = "ps4_data/data/protT5/output/per_residue_embeddings"


def generte_chunk_pt_embs(one_chunk=False):
    pth = PROTTRANS_EMBS_PATH
    for p in os.listdir(pth):
        if p != '.DS_Store':
            yield np.load(os.path.join(pth, p), allow_pickle=True)
            if one_chunk:
                break


def ss_tokeniser(ss, reverse=False):

    ss_set = ['C', 'T', 'G', 'H', 'S', 'B', 'I', 'E', 'C']

    if reverse:
        return inverse_ss_tokeniser(ss)
    else:
        return 0 if (ss == 'P' or ss == ' ') else ss_set.index(ss)


def inverse_ss_tokeniser(ss):

    ss_set = ['C', 'T', 'G', 'H', 'S', 'B', 'I', 'E', 'C', 'C']

    return ss_set[ss]


def res_tokeniser(res, reverse=False):

    residues = [
        "ALA", "ARG", "ASN", "ASP", "CYS",
        "GLU", "GLN", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO",
        "SER", "THR", "TRP", "TYR", "VAL",
        "SEC", "PYL", "UNK"
    ]

    if reverse:
        return residues[res]
    else:
        return residues.index(res)


def res_one_letter_to_three(letter, reverse=False):

    res_dict = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU",  "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER",  "T": "THR",  "W": "TRP", "Y": "TYR", "V": "VAL",
        "X": "UNK", "B": "UNK", "U": "UNK", "Z": "UNK"
    }

    if reverse:
        for one, three in res_dict.items():
            if three == letter:
                return one
    else:
        return res_dict[letter]


def q8_to_q3(q8):

    q3_arr = [0, 0, 1, 1, 0, 2, 1, 2]

    return q3_arr[q8] + 8


# Mark: main processing
def save_and_tokenise_data():

    df = pd.read_csv('ps4_data/data/data.csv')

    residues_t = []
    residues_v = []
    ss_t = []
    ss_v = []
    id_t = []
    id_v = []

    random.seed(37)

    for row in range(len(df["chain_id"])):

        chain_id = df['chain_id'][row]

        print(chain_id)
        residues_arr = []
        ss_arr = []

        ss_dict = {}

        # ss (part 1)
        ss_count = df['first_res'][row]
        for ss in df['dssp8'][row]:
            ss_dict[ss_count] = ss_tokeniser(ss)
            ss_count += 1

        # residues
        current_res = -99999

        res_count = df['first_res'][row]
        for r in df['input'][row]:

            if res_count == current_res:
                continue
            else:
                current_res = res_count

            res = 'C' if r.islower() else r
            res_three_letter = res_one_letter_to_three(res)
            residues_arr.append(res_tokeniser(res_three_letter))

            # ss (part 2)
            if res_count in ss_dict:
                ss_arr.append(ss_dict[res_count])
            else:
                print('missing residue:', chain_id, res_count)
                ss_arr.append(0)

            res_count += 1

        if len(residues_arr) > (2 ** 14):
            continue

        if len(residues_arr) != len(ss_arr):
            raise AssertionError(f"res arr length is {len(residues_arr)}, ss_arr length is "
                                 f"{len(ss_arr)}, should be equal")

        if random.random() > 0.95:
            residues_v.append(residues_arr)
            ss_v.append(ss_arr)
            id_v.append(chain_id)
        else:
            residues_t.append(residues_arr)
            ss_t.append(ss_arr)
            id_t.append(chain_id)

    np.savez_compressed('ps4_data/data/residues.npz', train=residues_t, valid=residues_v)
    np.savez_compressed('ps4_data/data/ss.npz', train=ss_t, valid=ss_v)
    np.savez_compressed('ps4_data/data/chain_ids.npz', train=id_t, valid=id_v)


def save_torch_data(d_type):

    all_d = np.load(f'ps4_data/data/{d_type}.npz', allow_pickle=True)
    for phase in ['train', 'valid']:
        for sample in all_d[phase]:
            converted_sample = torch.Tensor(sample).int()
            num = len(os.listdir(f'ps4_data/data/{phase}/{d_type}'))
            torch.save(converted_sample, f'ps4_data/data/{phase}/{d_type}/{num}.pt')


def save_pt_embs_torch():
    all_chains = np.load(f'ps4_data/data/chain_ids.npz', allow_pickle=True)
    for phase in ['train', 'valid']:
        print(phase)
        for chain in all_chains[phase]:
            print('\t', chain)
            for chunk in generte_chunk_pt_embs():
                if chain in chunk:
                    converted_sample = torch.Tensor(chunk[chain]).float()
                    print('\t\t', converted_sample.shape)
                    num = len(os.listdir(f'ps4_data/data/{phase}/prot_embs'))
                    torch.save(converted_sample, f'ps4_data/data/{phase}/prot_embs/{num}.pt')
                    break


def convert_to_protrans():
    for phase in ['train', 'valid']:
        all_d = np.load(f'ps4_data/data/residues.npz', allow_pickle=True)
        for sample in all_d[phase]:
            converted = __tokenise_as_protrans(sample)
            converted_sample = torch.Tensor(converted).int()
            num = len(os.listdir(f'ps4_data/data/{phase}/res_protrans'))
            torch.save(converted_sample, f'ps4_data/data/{phase}/res_protrans/{num}.pt')


def load_data(phase, shuffle=True):

    res_dir = 'prot_embs'

    lst = os.listdir(f'ps4_data/data/{phase}/{res_dir}')

    if shuffle:
        lst = random.sample(lst, len(lst))

    for file in lst:

        if not file.endswith(".pt"):
            continue

        R = torch.load(f'ps4_data/data/{phase}/{res_dir}/{file}')
        Y_ss = torch.load(f'ps4_data/data/{phase}/ss/{file}')

        yield R, Y_ss


def pt_2_csv(phase):

    inps = []
    dssps = []

    for r, c, yc, ys in load_data(phase, shuffle=False):

        input_str = ''
        dssp_str = ''

        for i in range(len(r)):
            res = r[i].item()
            res = res_tokeniser(res, reverse=True)
            res = res_one_letter_to_three(res, reverse=True)
            input_str += f'{res} '

            ss = ss_tokeniser(ys[i], reverse=True)
            dssp_str += f'{ss} '

        print(input_str)
        print(dssp_str)
        print(' ')

        inps.append(input_str)
        dssps.append(dssp_str)

    df = pd.DataFrame({
        'input': inps,
        ' dssp8': dssps
    })

    df.to_csv(fr'ps4_data/data/{phase}.csv', index=False, header=True)


# Mark: single sequence secondary structure
def load_cb513_dataset():

    df = pd.read_csv(f'ps4_data/data/cb513/CB513_HHblits.csv')
    embs = np.load(f'ps4_data/data/cb513/CB513_embeddings.npz', allow_pickle=True)
    for row in range(len(df)):
        res_string = df['input'][row]

        mask_str = ''
        ss_string = df[f' dssp8'][row]
        mask_raw = df[" cb513_mask"][row]
        mask_raw = mask_raw.split(" ")
        mask_raw = [int(float(x)) for x in mask_raw]
        for c in mask_raw:
            mask_str += f'{c} '

        _, y, mask = get_input_data_from_res_seq(res_string, ss_string, mask_str, 'CB513')
        r = torch.from_numpy(embs[str(row)]).float()
        yield r, y, mask


def get_input_data_from_res_seq(res_string, ss_string, mask_string, ds):

    R = []
    Y = []
    mask_out = []

    for i in range(len(res_string)):
        r_c = res_string[i]
        ss_c = ss_string[i]
        m_c = mask_string[i]
        if r_c != ' ':
            ss_c_val = ' ' if ss_c == 'C' else ss_c
            r_c_val = res_one_letter_to_three(r_c)
            R.append(res_tokeniser(r_c_val))
            Y.append(ss_tokeniser(ss_c_val))
            mask_out.append(m_c)

    return R, Y, mask_out


# MARK:- FASTA
def get_fasta_from_csv(csv_path, save_pth):

    df = pd.read_csv(csv_path)
    with open(save_pth, 'w') as f:
        for row in range(len(df["input"])):
            res_string = df['input'][row].replace(" ", "")
            header = f'>PS4-Extended|{row}'
            f.write(header + '\n')
            print(header)
            for i in range(0, len(res_string), 80):
                line = res_string[i:i+80]
                f.write(line + '\n')
                print(line)
            print(' ')
            f.write('\n')
        f.close()


def __tokenise_as_protrans(sample):

    protrans_tokeniser = [3, 8, 17, 10, 22, 9, 16, 5, 20, 12, 4, 14, 19, 15, 13, 7, 11, 21, 18, 6, -1, -2, 23]
    return [protrans_tokeniser[i] for i in sample]


