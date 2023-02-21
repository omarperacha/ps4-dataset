from transformers import T5EncoderModel, T5Tokenizer
import torch
import numpy as np
import time
import os
import wget


def generate_embedings(fasta_path):

    # Create directories
    protT5_path = "ps4_data/data/protT5"
    weights_path = "ps4_data/data/protT5/protT5_checkpoint"
    per_residue_path = "ps4_data/data/protT5/output/per_residue_embeddings"  # where to store the embeddings
    for dir_path in [protT5_path, weights_path, per_residue_path]:
        create_dir(dir_path)

    # Download weights
    weights_remote_url = "http://data.bioembeddings.com/public/embeddings/feature_models/t5/protT5_checkpoint.pt"
    _ = wget.download(weights_remote_url, weights_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using {}".format(device))

    # Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
    model, tokenizer = get_T5_model()

    # Load fasta.
    all_seqs = read_fasta(fasta_path)

    chunk_size = 1000

    # Compute embeddings and/or secondary structure predictions
    for i in range(0, len(all_seqs), chunk_size):
        keys = list(all_seqs.keys())[i: chunk_size + i]
        seqs = {k: all_seqs[k] for k in keys}
        results = get_embeddings(model, tokenizer, seqs, device)

        # Store per-residue embeddings
        save_embeddings(results["residue_embs"], per_residue_path + f"{i}.npz")


def get_T5_model(device):

    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

    return model, tokenizer


def read_fasta(fasta_path, split_char="|", id_field=1):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
        Returns dictionary holding multiple sequences or only single
        sequence, depending on input file.
    '''

    seqs = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # get pdb ID from header and create new entry
            if line.startswith('>'):
                pdb_id = line.replace('>', '').strip().split(split_char)
                pdb_id = pdb_id[id_field].lower() + pdb_id[id_field + 1]
                # replace tokens that are mis-interpreted when loading h5
                pdb_id = pdb_id.replace("/", "_").replace(".", "_")
                seqs[pdb_id] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines, drop gaps and cast to upper-case
                seq = ''.join(line.split()).upper().replace("-", "")
                # repl. all non-standard AAs and map them to unknown/X
                seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
                seqs[pdb_id] += seq
    example_id = next(iter(seqs))
    print("Read {} sequences.".format(len(seqs)))
    print("Example:\n{}\n{}".format(example_id, seqs[example_id]))

    return seqs


def save_embeddings(emb_dict,out_path):
    np.savez_compressed(out_path, **emb_dict)


def get_embeddings(model, tokenizer, seqs, device, per_residue=True,
                   max_residues=4000, max_seq_len=1000, max_batch=100):

    results = {"residue_embs": dict(),
               "protein_embs": dict(),
               "sec_structs": dict()
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            for batch_idx, identifier in enumerate(pdb_ids):  # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                if per_residue:  # store per-residue embeddings (Lx1024)
                    results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()
                    print("emb_count:", len(results["residue_embs"]))

    passed_time = time.time() - start
    avg_time = passed_time / len(results["residue_embs"]) if per_residue else passed_time / len(results["protein_embs"])
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time / 60, avg_time))
    print('\n############# END #############')
    return results


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)