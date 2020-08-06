# external libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import bidaf_layers


class BiDAFAttention(nn.Module):
    def __init__(self,
                 w_vocab_size, w_emb_size,
                 c_vocab_size, c_emb_size,
                 hidden_size, pretrained_embedding, drop_prob=0.):
        super(BiDAFAttention, self).__init__()
        self.emb = bidaf_layers.Embedding(w_vocab_size=w_vocab_size,
                                          w_emb_size=w_emb_size,
                                          c_vocab_size=c_vocab_size,
                                          c_emb_size=c_emb_size,
                                          hidden_size=hidden_size,
                                          drop_prob=drop_prob,
                                          pretrained_embedding=pretrained_embedding)

        self.enc = bidaf_layers.RNNEncoder(input_size=hidden_size * 2,
                                           hidden_size=hidden_size,
                                           num_layers=1,
                                           drop_prob=drop_prob)

        self.att = bidaf_layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                               drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)  # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)  # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)  # (batch_size, c_len, 8 * hidden_size)

        return att


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, drop_prob=0.1):
        super(SelfAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, c_mask):
        batch_size, c_len, _ = c.size()

        s = self.get_similarity_matrix(c, c)        # (batch_size, c_len, q_len)
        c1_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        c2_mask = c_mask.view(batch_size, 1, c_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, c2_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c1_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, c)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)
        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).
        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.
        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs