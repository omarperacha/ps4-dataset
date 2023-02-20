import time
import os
import math
from torch import save, cuda


def handle_cuda(unit, device, parallel):
    if cuda.is_available():
        if parallel:
            val = device
        else:
            val = 0
        return unit.to(f'cuda:{val}')
    else:
        return unit


def save_model_checkpoint(epoch, ave_loss_epoch, model, model_name, acc):
    test = os.listdir('models_ss/')

    for item in test:
        if item.endswith(".pt"):
            os.remove(os.path.join('models_ss', item))

    path_loss = round(ave_loss_epoch, 3)
    path_acc = '%.3f' % acc
    path = f'models_ss/{model_name}_epoch-{epoch}_loss-{path_loss}_acc-{path_acc}.pt'
    save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': path_loss
    }, path)
    print("\tSAVED MODEL TO:", path)


def time_since(t):
    now = time.time()
    s = now - t
    return '%s' % (__as_minutes(s))


def __as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)