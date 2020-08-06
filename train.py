import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from dataloader import data_loader
from models.bidaf_imple.model import BiDAF
from models.docqa_imple.model import DocQA

from trainer import Trainer
from evaluation import f1_score

def get_model(name):
    if name.lower() == "bidaf":
        _w_vocab_size = 10000
        _w_emb_size = 100
        _c_vocab_size = 100
        _c_emb_size = 8
        _hidden_size = 100
        _drop_prob = 0.2
        _pretrained_embedding = "info/mecab-embedding.npy"
        return BiDAF(w_vocab_size=_w_vocab_size, w_emb_size=_w_emb_size,
                     c_vocab_size=_c_vocab_size, c_emb_size=_c_emb_size,
                     hidden_size=_hidden_size, drop_prob=_drop_prob,
                     pretrained_embedding=_pretrained_embedding)
    elif name.lower() == "docqa":
        _w_vocab_size = 10000
        _w_emb_size = 100
        _c_vocab_size = 100
        _c_emb_size = 8
        _hidden_size = 100
        _drop_prob = 0.2
        _pretrained_embedding = "info/mecab-embedding.npy"
        return DocQA(w_vocab_size=_w_vocab_size, w_emb_size=_w_emb_size,
                     c_vocab_size=_c_vocab_size, c_emb_size=_c_emb_size,
                     hidden_size=_hidden_size, drop_prob=_drop_prob,
                     pretrained_embedding=_pretrained_embedding)
    else:
        raise AssertionError()

def forward(data, model, loss_fn, device):
    cw_ids = data['cw_ids'].to(device)
    cc_ids = data['cc_ids'].to(device)
    qw_ids = data['qw_ids'].to(device)
    qc_ids = data['qc_ids'].to(device)

    mask = torch.zeros_like(cw_ids) != cw_ids
    mask = mask.type(torch.float32).to(device)

    answer_ids = data['answer_ids'].to(device)
    start_targets = answer_ids[:, 0]
    end_targets = answer_ids[:, 1]

    logits = model(cw_ids, cc_ids, qw_ids, qc_ids)
    start_logits, end_logits = logits
    start_logits, end_logits = start_logits.squeeze(), end_logits.squeeze()
    start_logits = (mask * start_logits) + (1 - mask) * _neg_inf
    end_logits = mask * end_logits + (1 - mask) * _neg_inf

    loss = loss_fn(start_logits, start_targets) + loss_fn(end_logits, end_targets)
    return logits, loss

def get_metric(logits, data):
    start_logits, end_logits = logits
    start_logits, end_logits = start_logits.squeeze().cpu(), end_logits.squeeze().cpu()

    answers = data['answer']
    contexts = data['context']
    offsets = data['offset_mapping']
    cw_ids = data['cw_ids']

    mask = torch.zeros_like(cw_ids) != cw_ids
    mask = mask.type(torch.float32)

    start_logits = mask * start_logits + (1 - mask) * _neg_inf
    end_logits = mask * end_logits + (1 - mask) * _neg_inf
    pred_starts = torch.argmax(start_logits, dim=1)
    pred_ends = torch.argmax(end_logits, dim=1)

    f1 = 0
    n_samples = len(answers)
    for i in range(n_samples):
        true_answer = answers[i]

        start_token_idx, end_token_idx = pred_starts[i], pred_ends[i]
        start_char_idx, end_char_idx = offsets[i][start_token_idx][0], offsets[i][end_token_idx][1]
        pred_answer = contexts[i][start_char_idx:end_char_idx+1]
        f1 += f1_score(pred_answer, true_answer)
    f1_result = 100.0 * f1 / n_samples
    return f1_result


if __name__ == "__main__":
    import argparse

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    _root = './data'
    _neg_inf = -1e10

    _word_vectors = None
    _char_vectors = None

    _context_max_len = 600
    _context_word_len = 25
    _query_max_len = 60
    _query_word_len = 25

    _step_size = 1
    _gamma = 0.9

    args = argparse.ArgumentParser()
    args.add_argument("--lr", type=float)
    args.add_argument("--epochs", type=int)
    args.add_argument("--batch", type=int)
    args.add_argument("--print_iter", type=int)
    args.add_argument("--model_type", type=str)
    args.add_argument("--save_model_dir", type=str)
    args.add_argument("--seed", type=int)

    config = args.parse_args()

    _lr = config.lr
    _epochs = config.epochs
    _batch_size = config.batch
    _print_iter = config.print_iter
    _model_type = config.model_type
    _save_model_dir = config.save_model_dir
    _seed = config.seed

    _run_name = f"M-{_model_type}-step_size-{_step_size}-gamma-{_gamma}-BSZ-{_batch_size}-LR-{_lr}"
    print(_run_name)
    torch.manual_seed(_seed)

    train_loader = data_loader(_root, "train",
                               _context_max_len, _context_word_len,
                               _query_max_len, _query_word_len, batch_size=_batch_size)
    val_loader = data_loader(_root, "validate",
                             _context_max_len, _context_word_len,
                             _query_max_len, _query_word_len, batch_size=_batch_size)

    model = get_model(_model_type)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam([param for param in model.parameters() if param.requires_grad],
                     lr=_lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=_step_size, gamma=_gamma)
    trainer = Trainer(train_loader, val_loader,
                      model, loss_fn, optimizer, scheduler,
                      forward, get_metric,
                      device)
    trainer.train(_epochs, _batch_size, _print_iter, _save_model_dir, _run_name)

"""
python3 train.py --epoch=15 --batch=16 --lr=5e-3 \
--print_iter=400 --save_model_dir=saved_model --model_type=BiDAF \
--seed=123

python3 train.py --epoch=15 --batch=16 --lr=1e-3 \
--print_iter=400 --save_model_dir=saved_model --model_type=docqa \
--seed=123
"""
