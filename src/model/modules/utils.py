import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_rnn_hidden_states(batch_size, hidden_dim, n_layers, bidirectional, 
    rnn_type, init_type="zero"):
    n_directions = 2 if bidirectional else 1
    hidden_state_size = (
        n_directions*n_layers,
        batch_size,
        hidden_dim
    )

    def init_vec(size, init_type):
        if init_type == "zero":
            return torch.FloatTensor(*size).zero_().to(DEVICE)
        elif init_type == "uniform":
            return torch.FloatTensor(*size).uniform_(-1.0, 1.0).to(DEVICE)

    if rnn_type == "lstm":
        hiddens = (
            init_vec(hidden_state_size, init_type),
            init_vec(hidden_state_size, init_type),
        )
    else:
        hiddens = init_vec(hidden_state_size, init_type)
    return hiddens

def init_module_weights(m, init_w=0.08):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-1.0*init_w, init_w)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Embedding):
        m.weight.data.uniform_(-1.0*init_w, init_w)
    elif isinstance(m, nn.GRU) or isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
    else:
        raise Exception("undefined initialization for module {}".format(m))

def embedded_dropout(embed, words, dropout, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight, padding_idx, embed.max_norm, embed.norm_type, embed.scale_grad_by_freq, embed.sparse)
    return X

def gaussian_kld(mu1, var1,
                mu2=torch.FloatTensor([0.0]).to(DEVICE),
                var2=torch.FloatTensor([1.0]).to(DEVICE)):
    one = torch.FloatTensor([1.0]).to(DEVICE)
    return torch.sum(0.5 * 
                (torch.log(var2) 
                    - torch.log(var1)
                    + (var1 + (mu1 - mu2).pow(2)) / var2 - one
                ), 
                dim=1
            )
