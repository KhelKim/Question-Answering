import torch
from tqdm import tqdm

from train import get_model, forward
from dataloader import data_loader
from utils import save_json


def parse_model_path(path):
    model_info = path.split("/")[-1]
    model_name = model_info.split("-")[1]
    return model_info, model_name


if __name__ == "__main__":
    import argparse

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    _root = './data'
    _batch_size = 4
    _neg_inf = -1e10

    _context_max_len = 600
    _context_word_len = 25
    _query_max_len = 60
    _query_word_len = 25

    # args = argparse.ArgumentParser()
    # args.add_argument("--model_path", type=str)
    # config = args.parse_args()
    # _model_path = config.model_path

    _model_path = 'saved_model/M-BiDAF-step_size-2-gamma-0.8-BSZ-2-LR-0.005.pth'

    model_info, model_type = parse_model_path(_model_path)

    state = torch.load(_model_path)
    model = get_model(model_type)
    model.load_state_dict(state["model"])

    test_loader = data_loader(_root, "test",
                              _context_max_len, _context_word_len,
                              _query_max_len, _query_word_len, batch_size=_batch_size)

    pred_answers = {}
    for iter_, data in tqdm(enumerate(test_loader)):
        cw_ids = data['cw_ids'].to(device)
        cc_ids = data['cc_ids'].to(device)
        qw_ids = data['qw_ids'].to(device)
        qc_ids = data['qc_ids'].to(device)

        mask = torch.zeros_like(cw_ids) != cw_ids
        mask = mask.type(torch.float32).to(device)

        logits = model(cw_ids, cc_ids, qw_ids, qc_ids)
        start_logits, end_logits = logits
        start_logits, end_logits = start_logits.squeeze(), end_logits.squeeze()
        start_logits = (mask * start_logits) + (1 - mask) * _neg_inf
        end_logits = mask * end_logits + (1 - mask) * _neg_inf

        contexts = data['context']
        qids = data['qid']
        offsets = data['offset_mapping']
        pred_starts = torch.argmax(start_logits, dim=1)
        pred_ends = torch.argmax(end_logits, dim=1)

        for i in range(len(qids)):
            qid = qids[i]
            start_token_idx, end_token_idx = pred_starts[i], pred_ends[i]
            start_char_idx, end_char_idx = offsets[i][start_token_idx][0], offsets[i][end_token_idx][1]
            pred_answer = contexts[i][start_char_idx:end_char_idx + 1]

            pred_answers[qid] = pred_answer
        # break
    save_json(f"predictions/{model_info}_predict.json", pred_answers)
    print(pred_answers)
