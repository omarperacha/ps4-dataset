import torch

from data_ss.utils import res_tokeniser, atom_tokeniser, res_atoms, COORD_NORM, COORD_CLASSES
from data_ss.utils import load_data, load_ss_dataset
from torch import nn, cuda, load, device
from models_ss.ss_unit import MEGASSUnit, PTConvNet
from train_ss.losses import cat2cont
from edlib import align


def seq2pdb(atoms, residues, coords, guess_specific_atom=False, save=True):

    lines = ["MODEL        1"]
    current_res = ""
    res_atoms_list = []
    for i in range(len(atoms)):

        line_arr = [" "]*78
        line_arr[0:4] = ["A", "T", "O", "M"]

        atom_ser = ''.join((reversed(f"{i}")))
        for j in range(len(atom_ser)):
            line_arr[10-j] = atom_ser[j]

        atom = atom_tokeniser(atoms[i].item(), reverse=True)
        line_arr[77] = atom

        res = res_tokeniser(residues[i].item(), reverse=True)
        for r in range(3):
            line_arr[17 + r] = res[r]

        if res != current_res:
            current_res = res
            res_atoms_list = res_atoms(res)

        if guess_specific_atom:
            specific_atom = ''
            while specific_atom == '':
                for el in res_atoms_list:
                    if el[0] == atom:
                        specific_atom = el
                        res_atoms_list.remove(el)
                        break
                if specific_atom == '':
                    res_atoms_list = res_atoms(res)

            specific_atom = ''.join(reversed(specific_atom))
            for l in range(4):
                if len(specific_atom) > l:
                    line_arr[15 - l] = specific_atom[l]

        x_coord = ''.join((reversed(f"{round(cat2cont(coords[i * 3].item()), 3)}")))
        y_coord = ''.join((reversed(f"{round(cat2cont(coords[(i * 3) + 1].item()), 3)}")))
        z_coord = ''.join((reversed(f"{round(cat2cont(coords[(i * 3) + 2].item()), 3)}")))

        x_coord = "0" + x_coord if x_coord[2] == "." else x_coord
        y_coord = "0" + y_coord if y_coord[2] == "." else y_coord
        z_coord = "0" + z_coord if z_coord[2] == "." else z_coord
        x_coord = "00" + x_coord if x_coord[1] == "." else x_coord
        y_coord = "00" + y_coord if y_coord[1] == "." else y_coord
        z_coord = "00" + z_coord if z_coord[1] == "." else z_coord
        x_coord = "000." + x_coord if "." not in x_coord else x_coord
        y_coord = "000." + y_coord if "." not in y_coord else y_coord
        z_coord = "000." + z_coord if "." not in z_coord else z_coord

        for k in range(8):
            if k < len(x_coord):
                line_arr[37 - k] = x_coord[k]
            if k < len(y_coord):
                line_arr[45 - k] = y_coord[k]
            if k < len(z_coord):
                line_arr[53 - k] = z_coord[k]

        line = ''.join(line_arr)
        lines.append(line)

    lines.append("TER")
    lines.append("ENDMDL")

    if save:
        textfile = open("./eval/protein.pdb", "w")
        for l in lines:
            textfile.write(l + "\n")
        textfile.close()
    else:
        for l in lines:
            print(l)


def load_trained_model(load_path):

    model: nn.Module = MEGASSUnit()

    if load_path != '':
        try:
            if cuda.is_available():
                model.load_state_dict(load(load_path)['model_state_dict'])
            else:
                model.load_state_dict(load(load_path, map_location=device('cpu'))['model_state_dict'])
            print("loded params from", load_path)
        except:
            raise ImportError(f'No file located at {load_path}, could not load parameters')
    print(model)

    pytorch_total_params = sum(par.numel() for par in model.parameters() if par.requires_grad)
    print(pytorch_total_params)

    return model


def cat2cont(v):

    val = v % (COORD_CLASSES / 3)
    val = val / 4

    return val - COORD_NORM


def check_edit_distance(in_set='cb513', compare_set='train'):

    distances = []

    for in_seq, _, _, _ in __get_loader_for(in_set):
        min_pct = 100
        i_seq = []
        if in_set not in ['train', 'valid']:
            in_seq = torch.Tensor(in_seq)
        for ri in in_seq:
            i_seq.append(chr(int(ri.item()) + 161))
        for compare_seq, _, _, _ in __get_loader_for(compare_set):
            c_seq = []
            if compare_set not in ['train', 'valid']:
                compare_seq = torch.Tensor(compare_seq)
            for rs in compare_seq:
                c_seq.append(chr(int(rs) + 161))
            ed = align(i_seq, c_seq, task='distance')['editDistance']
            pct = ed / len(i_seq) * 100
            if min_pct > pct >= 0:
                min_pct = pct
        distances.append(min_pct)
        print(min_pct)
    print(min(distances), sum(distances)/len(distances))


def __get_loader_for(ds):

    dataset = ds.lower()

    loader_dict = {
        'train': load_data('train'),
        'valid': load_data('valid'),
        'cb513': load_ss_dataset('CB513', 8),
        'casp12': load_ss_dataset('CASP12', 8),
        'ts115': load_ss_dataset('TS115', 8),
        'new364': load_ss_dataset('NEW364', 8),
        'netsurf': load_ss_dataset('Train', 8),
    }

    return loader_dict[dataset]
