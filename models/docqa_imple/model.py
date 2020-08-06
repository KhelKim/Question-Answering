import torch
import torch.nn as nn
import torch.nn.functional as F

from . import bidaf_layers
from . import layers


class DocQA(nn.Module):
    def __init__(self,
                 w_vocab_size, w_emb_size,
                 c_vocab_size, c_emb_size,
                 hidden_size, pretrained_embedding, drop_prob=0.):
        super(DocQA, self).__init__()
        self.drop_prob = drop_prob

        self.bidaf = layers.BiDAFAttention(
            w_vocab_size=w_vocab_size, w_emb_size=w_emb_size,
            c_vocab_size=c_vocab_size, c_emb_size=c_emb_size,
            hidden_size=hidden_size, drop_prob=drop_prob,
            pretrained_embedding=pretrained_embedding
        )

        self.bidaf_proj = nn.Linear(8*hidden_size, hidden_size, bias=False)

        self.mod_rnn = bidaf_layers.RNNEncoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            drop_prob=drop_prob
        )

        self.self_att = layers.SelfAttention(
            hidden_size=2*hidden_size,
            drop_prob=drop_prob
        )

        self.self_att_proj = nn.Linear(8*hidden_size, hidden_size, bias=False)

        self.start_rnn = bidaf_layers.RNNEncoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            drop_prob=drop_prob
        )

        self.start_linear = nn.Linear(2*hidden_size, 1)

        self.end_rnn = bidaf_layers.RNNEncoder(
            input_size=3*hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            drop_prob=drop_prob
        )

        self.end_linear = nn.Linear(2*hidden_size, 1)

    def forward(self, cw_ids, cc_ids, qw_ids, qc_ids):
        c_mask = torch.zeros_like(cw_ids) != cw_ids
        c_len = c_mask.sum(-1)

        ################## bidaf
        c_att = self.bidaf(cw_ids, cc_ids, qw_ids, qc_ids)  # (batch, context_len, 8*hidden)
        c_att = self.bidaf_proj(c_att)  # (batch, context_len, hidden)

        ###################### mod rnn encoding + self atten
        c_att = F.dropout(c_att, p=self.drop_prob)
        c_enc = self.mod_rnn(c_att, c_len)  # (batch, context_len, 2*hidden)
        c_enc = F.dropout(c_enc, p=self.drop_prob)
        c_s_att = self.self_att(c_enc, c_mask)  # (batch, context_len, 8*hidden)
        c_s_att = self.self_att_proj(c_s_att)  # (batch, context_len, hidden)

        #################################
        c_final = c_att + c_s_att

        #################################
        c_final = F.dropout(c_final, p=self.drop_prob)
        start_input = c_final
        start_enc = self.start_rnn(start_input, c_len)
        end_input = torch.cat([start_enc, c_final], dim=-1)

        #################################
        start_enc = F.dropout(start_enc, p=self.drop_prob)
        start_logits = self.start_linear(start_enc)

        #################################
        end_enc = self.end_rnn(end_input, c_len)
        end_logits = self.end_linear(end_enc)

        return start_logits, end_logits
