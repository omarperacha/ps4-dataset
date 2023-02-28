import sys
from ps4_train.train_ps4 import *
from ps4_data.get_embeddings import generate_embedings
from ps4_eval.eval import *


def print_help():
    print('Usage: python main.py [option] [args]')
    print('Options:')
    print('\t--train, -t\t\tTrain a model')
    print('\t\tArguments:')
    print('\t\t\t--mega\t\t\tUse the Mega model')
    print('\t\t\t--conv\t\t\tUse the Conv model')

    print('\t--eval, -e\t\tEvaluate a model')
    print('\t\tOrdered Arguments:')
    print('\t\t\t1: dataset to evaluate on')
    print('\t\t\t--cb513\t\t\tEvaluate on CB513')
    print('\t\t\t--ps4\t\t\tEvaluate on PS4')
    print('\t\t\t2: model to use')
    print('\t\t\t--mega\t\t\tUse the Mega model')
    print('\t\t\t--conv\t\t\tUse the Conv model')
    print('\t\t\t3: path to model weights - optional')
    print('\t\t\te.g. ps4_models/Mega/PS4-Mega_loss-0.633_acc-78.176.pt')

    print('\t--gen_dataset, -gd\tGenerate a dataset for training')


if __name__ == '__main__':

    if len(sys.argv) > 1:
        if sys.argv[1] in ['--train', '-t']:
            if len(sys.argv) > 2 and sys.argv[2] == '--mega':
                train(31, 'PS4_Mega')
            elif len(sys.argv) > 2 and sys.argv[2] == '--conv':
                train(31, 'PS4_Conv')
            else:
                train(31, 'PS4_Mega')

        elif sys.argv[1] in ['--eval', '-e']:
            if len(sys.argv) > 2:
                model_name = 'PS4_Mega'
                path = 'ps4_models/Mega/PS4-Mega_loss-0.633_acc-78.176.pt'

                if len(sys.argv) > 3:
                    if sys.argv[3] in ['--mega', '--conv']:
                        model_name = 'PS4_Mega' if sys.argv[3] == '--mega' else 'PS4_Conv'
                    else:
                        print(f'Please specify a valid model name. found: {sys.argv[3]}')

                    if len(sys.argv) > 4:
                        path = sys.argv[4]

                if sys.argv[2] == '--cb513':
                    eval_cb513(path, model_name=model_name)
                elif sys.argv[2] == '--ps4':
                    eval_ps4_test(path, model_name=model_name)
                else:
                    print(f'Please specify a valid evaluation dataset. found: {sys.argv[2]}')

        elif sys.argv[1] in ['--gen_dataset', '-gd']:
            generate_embedings('ps4_data/ps4_data/data.fasta')
            save_and_tokenise_data()
            save_pt_embs_torch()
            for dtype in ['residues', 'ss']:
                save_torch_data(dtype)

        else:
            print_help()

    else:
        print_help()





