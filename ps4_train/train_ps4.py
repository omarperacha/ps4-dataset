import time
import torch
from torch import nn, optim, device, cuda, set_grad_enabled, load, sum
from torch.nn.utils import clip_grad_norm_
from ps4_data.utils import TRAIN_SAMPLES, NUM_SAMPLES, load_data
from ps4_models.classifiers import PS4_Conv, PS4_Mega
from ps4_train.utils import save_model_checkpoint, time_since
from ps4_train.losses import CrossEntropyTimeDistributedLoss
from builtins import sum as osum


CV_PHASES = ['train', 'valid']
TRAIN_ONLY_PHASES = ['train']


def train(
        epochs,
        save_model=True,
        load_path='',
        shuffle_batches=True,
        num_btches=NUM_SAMPLES,
        val=True,
        evaluate=False
):

    loss_config = {
        "batch_size": 1
    }

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

    pytorch_total_params = osum(par.numel() for par in model.parameters() if par.requires_grad)
    print(pytorch_total_params)

    if cuda.is_available():
        # setting model and data devices is handled within model files, allowing 4 GPU model distributed training
        torch.cuda.empty_cache()

    base_lr = 0.6
    max_lr = 0.00003

    batch_size = loss_config["batch_size"]

    train_samples = TRAIN_SAMPLES
    num_batches = num_btches

    optimiser = optim.Adam([x for x in model.parameters() if x.requires_grad], base_lr)

    cath_criterion = CrossEntropyTimeDistributedLoss()

    scheduler = optim.lr_scheduler.OneCycleLR(optimiser, max_lr,
                                              epochs=epochs, steps_per_epoch=num_batches, pct_start=0.1,
                                              anneal_strategy='cos', cycle_momentum=True, base_momentum=0.8,
                                              max_momentum=0.95, div_factor=1000.0, final_div_factor=1000.0,
                                              last_epoch=-1)

    best_val_acc = 0.0

    if val:
        phases = CV_PHASES
    elif evaluate:
        phases = ["valid"]
    else:
        phases = TRAIN_ONLY_PHASES

    for epoch in range(epochs):
        start = time.time()
        pr_interval = 200 if not evaluate else 1

        print(f'Beginning EPOCH {epoch + 1}')

        for phase in phases:

            count = 0
            batch_count = 0
            token_count = 0
            loss_epoch = 0
            running_accuray = 0.0
            running_token_count = 0
            print_loss_batch = 0  # Reset on print
            print_acc_batch = 0  # Reset on print

            print(f'\n\tPHASE: {phase}')

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            model.zero_grad()

            for r, ys in load_data(phase, shuffle=shuffle_batches, protrans=True):

                batch_count += 1

                seq_size = len(r)
                token_count += seq_size
                running_token_count += seq_size

                Ys = ys.view(1, seq_size).long()

                R = r.view(1, seq_size, 1024)

                with set_grad_enabled(phase == 'train'):
                    y_hat = model(R)
                    _, ss_preds = torch.max(torch.softmax(y_hat, 2), 2)

                    loss = cath_criterion(y_hat, Ys.to(y_hat.device))

                loss_epoch += loss.item()

                print_loss_batch += loss.item()

                running_accuray += sum(ss_preds.cpu() == Ys.cpu())
                print_acc_batch += sum(ss_preds.cpu() == Ys.cpu())

                count += 1
                if phase == 'train':
                    loss.backward()

                if phase == 'train' and (batch_count == batch_size or count == train_samples):
                    clip_grad_norm_(model.parameters(), 2.0)
                    optimiser.step()
                    scheduler.step()

                    batch_count = 0

                    model.zero_grad()

                # print loss for recent set of batches
                if count % pr_interval == 0:
                    ave_loss = print_loss_batch/pr_interval
                    ave_acc = 100 * print_acc_batch.float()/running_token_count

                    print_acc_batch = 0
                    running_token_count = 0

                    print('\t\t[%d] loss: %.3f, s_acc: %.3f, LR: %.7f' % (
                        count, ave_loss, ave_acc, optimiser.state_dict()["param_groups"][0]["lr"]
                    ))
                    print_loss_batch = 0

                stop = TRAIN_SAMPLES
                if count == stop:
                    break

            # calculate loss and accuracy for phase
            ave_loss_epoch = loss_epoch/count

            epoch_acc = 100 * running_accuray.float() / token_count
            print('\tfinished %s phase [%d] loss: %.3f, s_acc: %.3f' % (phase,
                                                                        epoch + 1,
                                                                        ave_loss_epoch,
                                                                        epoch_acc))

        print('\n\ttime:', time_since(start), '\n')

        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            print("\tNEW BEST s_acc: %.3f" % best_val_acc, '\n')

            # save model when validation loss improves
            if save_model:
                save_model_checkpoint(epoch, ave_loss_epoch, model, f"OmegaFold", epoch_acc)

        else:
            print("\ts_acc DID NOT IMPROVE FROM %.3f" % best_val_acc, '\n')

    print("DONE")



