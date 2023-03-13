from ps4_data.utils import load_data, load_alt_dataset
from torch import nn, cuda, load, device
from ps4_models.classifiers import PS4_Mega, PS4_Conv


def load_trained_model(load_path, model_name='PS4_Mega'):

    if model_name.lower() not in ['ps4_conv', 'ps4_mega']:
        raise ValueError(f'Model name {model_name} not recognised, please choose from PS4_Conv, PS4_Mega')

    model: nn.Module = PS4_Mega() if model_name.lower() == 'ps4_mega' else PS4_Conv()

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


def __get_loader_for(ds):

    dataset = ds.lower()

    loader_dict = {
        'train': load_data('train'),
        'valid': load_data('valid'),
        'cb513': load_alt_dataset('cb513'),
        'ts115': load_alt_dataset('ts115')
    }

    return loader_dict[dataset]
