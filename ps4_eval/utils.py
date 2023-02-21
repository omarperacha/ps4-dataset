import torch

from ps4_data.utils import load_data, load_cb513_dataset
from torch import nn, cuda, load, device
from ps4_models.classifiers import PS4_Mega, PS4_Conv
from edlib import align


def load_trained_model(load_path):

    model: nn.Module = PS4_Mega()

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
        'cb513': load_cb513_dataset()
    }

    return loader_dict[dataset]
