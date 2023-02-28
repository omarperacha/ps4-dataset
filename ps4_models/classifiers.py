import torch
import torch.nn as nn
from ps4_data.utils import SS_CLASSES
from mega.fairseq.modules.mega_layer import MegaEncoderLayer


class PS4_Conv(torch.nn.Module):
    def __init__(self):
        super(PS4_Conv, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 512, kernel_size=(7, 1), padding=(3, 0)),  # 7x512
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(512, 256, kernel_size=(7, 1), padding=(3, 0)),  # 7x256
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(256, 128, kernel_size=(7, 1), padding=(3, 0)),  # 7x128
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(128, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        n_final_in = 32

        self.dssp8_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 8, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        x = self.elmo_feature_extractor(x)  # OUT: (B x 32 x L x 1)
        d8_yhat = self.dssp8_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 8)
        
        return d8_yhat


class PS4_Mega(nn.Module):
    def __init__(self, nb_layers=11, l_aux_dim=1024, model_parallel=False,
                 h_dim=1024, batch_size=1, seq_len=1, dropout=0.0):
        super(PS4_Mega, self).__init__()

        self.nb_layers = nb_layers
        self.h_dim = h_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dropout = dropout
        self.aux_emb_size_l = l_aux_dim
        self.input_size = l_aux_dim

        self.args = ArgHolder(emb_dim=self.input_size, dropout=dropout, hdim=h_dim)

        self.nb_tags = SS_CLASSES

        self.model_parallel = model_parallel

        # build actual NN
        self.__build_model()

    def __build_model(self):

        # design Sequence processing module

        megas = []
        for i in range(self.nb_layers):
            mega = MegaEncoderLayer(self.args)
            megas.append(mega)

        self.seq_unit = MegaSequence(*megas)

        self.dropout_i = nn.Dropout(max(0.0, self.dropout - 0.2))

        # output layer which projects back to tag space
        out_dim = self.input_size

        self.hidden_to_tag = nn.Linear(out_dim, self.nb_tags, bias=False)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_rnn_units)
        hidden_a = torch.randn(self.nb_rnn_layers, self.batch_size, self.aux_emb_size_l)

        if torch.cuda.is_available():
            hidden_a = hidden_a.cuda()

        return hidden_a

    def forward(self, r):

        self.seq_len = r.shape[1]

        # residue encoding
        R = r.view(self.seq_len, self.batch_size,  self.aux_emb_size_l)

        X = self.dropout_i(R)

        # Run through MEGA
        X = self.seq_unit(X, encoder_padding_mask=None)
        X = X.view(self.batch_size, self.seq_len, self.input_size)

        # run through linear layer
        X = self.hidden_to_tag(X)

        Y_hat = X
        return Y_hat


class MegaSequence(nn.Sequential):
    def forward(self, input, **kwargs):
        for module in self:
            options = kwargs if isinstance(module, MegaEncoderLayer) else {}
            input = module(input, **options)
        return input


class ArgHolder(object):
    def __init__(self, hdim=512, dropout=0.1, emb_dim=1024):
        super(object, self).__init__()

        self.encoder_embed_dim = emb_dim
        self.encoder_hidden_dim = hdim
        self.dropout = dropout
        self.encoder_ffn_embed_dim = 1024
        self.ffn_hidden_dim: int = 1024
        self.encoder_z_dim: int = 128
        self.encoder_n_dim: int = 16
        self.activation_fn: str = 'silu'
        self.attention_activation_fn: str = 'softmax'
        self.attention_dropout: float = 0.0
        self.activation_dropout: float = 0.0
        self.hidden_dropout: float = 0.0
        self.encoder_chunk_size: int = -1
        self.truncation_length: int = None
        self.rel_pos_bias: str = 'simple'
        self.max_source_positions: int = 2048
        self.normalization_type: str = 'layernorm'
        self.normalize_before: bool = False
        self.feature_dropout: bool = False



