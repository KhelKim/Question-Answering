import glob

import torch
from torch.utils import data

from utils import read_json

class CustomDataset(data.Dataset):
    def __init__(self, root, phase,
                 context_max_len, context_word_max_len,
                 query_max_len, query_word_max_len):
        self.root = root
        self.phase = phase
        self.context_max_len = context_max_len
        self.context_word_max_len = context_word_max_len
        self.query_max_len = query_max_len
        self.query_word_max_len = query_word_max_len
        self.file_list = glob.glob(f"{root}/{phase}/*.json")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        item = read_json(file_path)
        # item's keys
        # 'context', 'cw_ids', 'cc_ids', 'offset_mapping',
        # 'qid', 'q', 'qw_ids', 'qc_ids',
        # 'answer', 'answer_ids' (if phase != 'test')
        for key, value in item.items():
            if key in ['context', 'qid', 'answer', 'q']:
                item[key] = value
            elif key == "cw_ids":
                padded_ids = value[:self.context_max_len] + [0] * (self.context_max_len - len(value))
                item[key] = torch.LongTensor(padded_ids)
            elif key == "qw_ids":
                padded_ids = value[:self.query_max_len] + [0] * (self.query_max_len - len(value))
                item[key] = torch.LongTensor(padded_ids)
            elif key == "cc_ids":
                padded_ids = []
                for token in value:
                    word_padded_ids = token[:self.context_word_max_len] + [0] * (self.context_word_max_len - len(token))
                    padded_ids.append(word_padded_ids)
                padded_ids = padded_ids[:self.context_max_len]
                for _ in range(max([self.context_max_len - len(padded_ids), 0])):
                    padded_ids.append([0]*self.context_word_max_len)
                item[key] = torch.LongTensor(padded_ids)
            elif key == "qc_ids":
                padded_ids = []
                for token in value:
                    word_padded_ids = token[:self.query_word_max_len] + [0] * (self.query_word_max_len - len(token))
                    padded_ids.append(word_padded_ids)
                padded_ids = padded_ids[:self.query_max_len]
                for _ in range(max([self.query_max_len - len(padded_ids), 0])):
                    padded_ids.append([0] * self.query_word_max_len)
                item[key] = torch.LongTensor(padded_ids)
            elif key == "answer_ids":
                start_token_idx, end_token_idx = value
                if self.context_max_len <= end_token_idx:
                    start_token_idx, end_token_idx = 0, 0
                item[key] = torch.LongTensor([start_token_idx, end_token_idx])
            elif key == "offset_mapping":
                padded_mapping = value[:self.context_max_len] + [(0, 0)] * (self.context_max_len - len(value))
                item[key] = torch.LongTensor(padded_mapping)
        return item


def data_loader(root, phase,
                context_max_len, context_word_len,
                query_max_len, query_word_len, batch_size=4):
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False
    dataset = CustomDataset(root, phase,
                            context_max_len, context_word_len,
                            query_max_len, query_word_len)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    _root = "data"
    _phase = "train"
    _context_max_len = 10
    _context_word_len = 10
    _query_max_len = 3
    _query_word_len = 10
    dset = CustomDataset(_root, _phase,
                         _context_max_len, _context_word_len,
                         _query_max_len, _query_word_len)
    # for res in dset:
    #     print(res)
    #     print(res.keys())
    #     break

    dataloader = data_loader(_root, _phase,
                             _context_max_len, _context_word_len,
                             _query_max_len, _query_word_len, batch_size=2)
    for res in dataloader:
        print(res)
        # print(res['offset_mapping'].size())
        print(res.keys())
        break

