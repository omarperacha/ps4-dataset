from ps4_eval.utils import *
from ps4_data.utils import *
import numpy as np


def eval_ps4_test(load_path, model_name='PS4_Mega'):
    print("starting")
    model = load_trained_model(load_path, model_name)

    ss_confusion = np.zeros((SS_CLASSES, SS_CLASSES))
    count = 0

    whole_accs = []
    q3_accs = []

    for r, ys in load_data('valid', shuffle=False):

        count += 1

        seq_size = len(r)

        Ys = ys.view(1, seq_size).long()

        R = r.view(1, seq_size)

        with torch.no_grad():
            y_hat = model(R)
            probs = torch.softmax(y_hat, 2)
            _, ss_preds = torch.max(probs, 2)

            ss_acc = 0
            q3_acc = 0

            for i in range(seq_size):
                val = Ys[0][i].item()
                cand = ss_preds[0][i].item()

                ss_confusion[val, cand] += 1

                ss_acc += 1 if val == cand else 0
                q3_acc += 1 if __assess_q3(val, cand) else 0

            ss_acc /= seq_size
            print("\n", count)
            print("accuracy:", ss_acc, "len:", seq_size)
            whole_accs.append(ss_acc)
            __q8_q3_from_confusion(ss_confusion)

    print(f"val acc: {sum(whole_accs)/len(whole_accs)}\n"
          f"q3 acc: {sum(q3_accs)/len(whole_accs)}")


def eval_alt(load_path, ds_name='cb513', use_mask=True, model_name='PS4_Mega'):
    print("starting")
    model = load_trained_model(load_path, model_name)

    ss_confusion = np.zeros((SS_CLASSES, SS_CLASSES))
    count = 0

    whole_accs = []
    val_accs = []
    q3_accs = []

    for r, y, mask in load_alt_dataset(ds_name):

        count += 1

        seq_size = len(r)

        R = r.view(1, seq_size, 1024)
        ys = torch.IntTensor(y)

        with torch.no_grad():
            y_hat = model(R)
            probs = torch.softmax(y_hat, 2)
            ss_probs, ss_preds = torch.max(probs, 2)

            ss_acc = 0
            val_acc = 0
            q3_acc = 0
            val_size = 0

            for i in range(seq_size):
                val = ys[i].item()
                cand = ss_preds[0][i].item()

                if val == cand:
                    ss_acc += 1

                if not use_mask or mask[i] == "1":
                    val_acc += 1 if val == cand else 0
                    q3_acc += 1 if __assess_q3(val, cand) else 0
                    val_size += 1
                    ss_confusion[val, cand] += 1

            ss_acc /= seq_size
            if val_size > 0:
                val_acc /= val_size
                q3_acc /= val_size
                val_accs.append(val_acc)
                q3_accs.append(q3_acc)
            print("\n", count)
            print("whole sequence q8 accuracy:", ss_acc, "len:", seq_size)
            whole_accs.append(ss_acc)
            __q8_q3_from_confusion(ss_confusion)

    print(f"\nDONE: {ds_name}, whole q8 acc: {sum(whole_accs)/len(whole_accs)}\n"
          f"val only q8 acc: {sum(val_accs)/len(val_accs)}\n"
          f"q3 acc: {sum(q3_accs)/len(val_accs)}")


# MARK: private
def __get_res_pred(current_r_probs, thresh=0):
    cummul_prob = torch.zeros_like(current_r_probs[0])
    atoms = 0
    for rp in current_r_probs:
        cummul_prob += rp
        atoms += 1
    cummul_prob /= atoms
    prob, res_pred = torch.max(cummul_prob, 0)

    post = res_pred.item() if prob > thresh else 0

    return post


# MARK: sampling from new sequence
def sample_new_sequence(embs_load_path, weights_load_path, model_name='PS4_Mega'):

    model = load_trained_model(weights_load_path, model_name)

    for r in load_embs_for_sampling(embs_load_path):

        seq_size = len(r)
        R = r.view(1, seq_size, 1024)

        pred_ss = ''

        with torch.no_grad():
            y_hat = model(R)
            probs = torch.softmax(y_hat, 2)
            _, ss_preds = torch.max(probs, 2)

            for i in range(seq_size):
                ss = ss_preds[0][i].item()
                ss = ss_tokeniser(ss, reverse=True)
                pred_ss += ss

        print(f'\t{pred_ss}')


def __q8_q3_from_confusion(confusion):

    # c, h, e
    q3 = np.zeros((3, 8))

    # Q8
    for i in range(len(confusion)):

        if i in [2, 3, 6]:
            q3[1] += confusion[i]
        elif i in [5, 7]:
            q3[2] += confusion[i]
        else:
            q3[0] += confusion[i]


def __assess_q3(val, cand):
    if val in [2, 3, 6]:
        return cand in [2, 3, 6]
    elif val in [5, 7]:
        return cand in [5, 7]
    else:
        return cand in [0, 1, 4]

