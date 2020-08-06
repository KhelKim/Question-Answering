import re
from collections import Counter
from konlpy.tag import Mecab
import hgtk
from tqdm import tqdm

from utils import read_text, read_json, save_json

hangul = re.compile(r'[^ㄱ-ㅣ가-힣 0-9]+')

def make_texts(data):
    texts = []
    for line in data:
        row = line.strip()
        texts.append(row)
    return texts

def preprocess_text(text):
    text = hangul.sub(" ", text)
    text = " ".join(text.split())
    return text

def get_offset_mapping(origin_text, words):
    origin_idx = 0
    offset_mapping = []
    for word in words:
        for i, c in enumerate(origin_text[origin_idx:]):
            if c == word[0]:
                start_idx = origin_idx + i
                end_idx = start_idx + len(word) - 1
                origin_idx = end_idx + 1

                offset_mapping.append((start_idx, end_idx))

                if word != origin_text[start_idx:end_idx+1]:
                    print("offset mapping이 잘 안됨")
                    print("\n다음은 토큰들")
                    print(words)
                    print("\n다음은 원문")
                    print(origin_text)
                    print('\n좌: 토큰, 우: offset으로 뽑은 값')
                    print(word, origin_text[start_idx:end_idx+1])
                    print("으악")
                break
    return offset_mapping

def get_start_end_token_idx(answer_start, answer_end, offset_mapping):
    start_token_idx, end_token_idx = None, None
    previous_end = -1
    for idx, (start, end) in enumerate(offset_mapping):
        if previous_end < answer_start <= end:
            start_token_idx = idx
        if answer_end <= start:
            end_token_idx = idx
            break
        elif start <= answer_end <= end:
            end_token_idx = idx
            break
        previous_end = end
    if start_token_idx is None:
        print(answer_start, answer_end)
        print(offset_mapping)
        raise AssertionError('something is wrong!(start_token_id)')
    elif end_token_idx is None:
        return start_token_idx, len(offset_mapping)-1
    return start_token_idx, end_token_idx


def preprocess_data(path, with_answer, save_path, w_tokenizer, c_tokenizer):
    new_data = []
    data = read_json(path)

    for topic in tqdm(data['data']):
        for item in topic['paragraphs']:  # len(topic['paragraphs']) == 1
            new_item = {}

            context = item['context']
            context_dict = w_tokenizer.encode_plus(context)
            cw_ids = context_dict['w_ids']
            offset_mapping = context_dict['offset_mapping']
            cc_ids = c_tokenizer.encode(context, w_tokenizer)

            new_qas = []
            qas = item['qas']
            for qa in qas:
                new_qa = {}
                qid = qa['id']
                q = qa['question']

                qw_ids = w_tokenizer.encode(q)
                qc_ids = c_tokenizer.encode(q, w_tokenizer)

                new_qa['id'] = qid
                new_qa['question'] = q
                new_qa['qw_ids'] = qw_ids
                new_qa['qc_ids'] = qc_ids
                if with_answer:
                    answer = qa['answers'][0]['text']
                    answer_start = qa['answers'][0]['answer_start']
                    answer_end = answer_start + len(answer) - 1
                    start_token_idx, end_token_idx = get_start_end_token_idx(answer_start, answer_end, offset_mapping)
                    new_qa['answer'] = answer
                    new_qa['answer_ids'] = [start_token_idx, end_token_idx]
                new_qas.append(new_qa)
            new_item['context'] = context
            new_item['cw_ids'] = cw_ids
            new_item['cc_ids'] = cc_ids
            new_item['offset_mapping'] = offset_mapping
            new_item['qas'] = new_qas

            new_data.append(new_item)
    save_json(save_path, new_data)


class CharTokenizer(object):
    @classmethod
    def make_vocab(cls, data_path, save_path):
        data = read_text(data_path)
        texts = make_texts(data)

        letters = []
        for text in tqdm(texts):
            text = preprocess_text(text)
            for char in text:
                try:
                    ls = hgtk.letter.decompose(char)
                except:
                    ls = ["[NUM]", "[NUM]", "[NUM]"]
                letters.extend(ls)
        letter_counter = Counter(letters)
        vocab = {"[PAD]": 0, "[UNK]": 1}
        idx = 2
        for char, count in letter_counter.most_common():
            vocab[char] = idx
            idx += 1

        save_json(save_path, vocab)

    def __init__(self, vocab_path, vocab_size):
        vocab = read_json(vocab_path)
        self.vocab = {key: value for key, value in vocab.items() if value < vocab_size}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.vocab.get("[PAD]", None)
        self.unk_token_id = self.vocab.get("[UNK]", None)

    def tokenize(self, text, w_tokenizer):
        c_tokens = []
        for token in w_tokenizer.tokenize(text):
            ls = []
            for char in token:
                try:
                    l = hgtk.letter.decompose(char)
                except:
                    l = ["[NUM]", "[NUM]", "[NUM]"]
                ls.append(l)
            c_tokens.append(ls)
        return c_tokens

    def encode(self, text, w_tokenizer):
        tokens = self.tokenize(text, w_tokenizer)
        ids = []
        for word in tokens:
            w_ids = [self.vocab.get(char, self.unk_token_id) for ls in word for char in ls]
            ids.append(w_ids)
        return ids

    def encode_plus(self, text):
        return AssertionError("char은 encode_plus 없음")


class MecabTokenizer(object):
    tokenizer = Mecab()

    @classmethod
    def make_vocab(cls, data_path, save_path):
        texts = read_text(data_path)
        words = [word for text in tqdm(texts) for word in cls.tokenizer.morphs(preprocess_text(text))]
        word_counter = Counter(words)

        vocab = {"[PAD]": 0, "[UNK]": 1}
        idx = 2
        for word, count in word_counter.most_common():
            vocab[word] = idx
            idx += 1
        save_json(save_path, vocab)

    def __init__(self, vocab_path, vocab_size):
        vocab = read_json(vocab_path)
        self.vocab = {key: value for key, value in vocab.items() if value < vocab_size}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = self.vocab.get("[PAD]", None)
        self.unk_token_id = self.vocab.get("[UNK]", None)

    def tokenize(self, text):
        text = preprocess_text(text)
        return self.tokenizer.morphs(text)

    def encode(self, text):
        text = preprocess_text(text)
        tokens = self.tokenize(text)
        ids = [self.vocab.get(token.strip(), self.unk_token_id) for token in tokens]
        return ids

    def encode_plus(self, text):
        preprocessed_text = preprocess_text(text)
        tokens = self.tokenize(preprocessed_text)
        ids = [self.vocab.get(token.strip(), self.unk_token_id) for token in tokens]
        offset = get_offset_mapping(text, tokens)
        return {"w_ids": ids, "offset_mapping": offset}


if __name__ == "__main__":
    _mecab_vocab_path = "info/mecab-vocab.json"
    _char_vocab_path = "info/char-vocab.json"

    _w_vocab_size = 10000
    _c_vocab_size = 100

    # CharTokenizer.make_vocab("info/train.txt", "info/char-vocab.json")
    # MecabTokenizer.make_vocab("info/train.txt", "info/mecab-vocab.json")

    w_tokenizer = MecabTokenizer(_mecab_vocab_path, _w_vocab_size)
    c_tokenizer = CharTokenizer(_char_vocab_path, _c_vocab_size)

    # sample_text = "[속보] 오늘 저녁은 순댓국 입니다."
    #
    # print(w_tokenizer.tokenize(sample_text))
    # print(w_tokenizer.encode(sample_text))
    # print(w_tokenizer.encode_plus(sample_text))
    #
    # print(c_tokenizer.tokenize(sample_text, w_tokenizer))
    # print(c_tokenizer.encode(sample_text, w_tokenizer))

    _path = 'data/validate.json'
    _with_answer = True
    _save_path = 'data/preprocessed_validate.json'
    preprocess_data(_path, _with_answer, _save_path, w_tokenizer, c_tokenizer)

    _path = 'data/train.json'
    _with_answer = True
    _save_path = 'data/preprocessed_train.json'
    preprocess_data(_path, _with_answer, _save_path, w_tokenizer, c_tokenizer)

    _path = 'data/test.json'
    _with_answer = False
    _save_path = 'data/preprocessed_test.json'
    preprocess_data(_path, _with_answer, _save_path, w_tokenizer, c_tokenizer)

    # sample_text = "유지아이지(더휴컴퍼니)이다. 또 폴햄(에이션패션), 지센(위비스), 지이크(신원), 숲(동광인터내셔날), 네파(평안섬유공업)"
    #
    # print(w_tokenizer.tokenize(sample_text))
    # print(w_tokenizer.encode(sample_text))
    # print(w_tokenizer.encode_plus(sample_text))