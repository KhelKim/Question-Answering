import os
import argparse
from utils import read_json


def make_txt(data):
    texts = []
    for topic in data['data']:
        text = []
        for item in topic['paragraphs']:
            context = item['context']
            qas = item['qas']
            text.append(context)
            for qa in qas:
                q = qa['question']
                text.append(q)
            texts.append(" ".join(text))
    return "\n".join(texts) + "\n"

def save_txt(path, text):
    directory = "/".join(path.split("/")[:-1])
    os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding="utf-8-sig") as f:
        f.write(text)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str)
    args.add_argument("--save_path", type=str)

    config = args.parse_args()

    data_path = config.data_path
    save_path = config.save_path

    data = read_json(data_path)
    text = make_txt(data)
    save_txt(save_path, text)

"""
python3 convert_json2txt.py --data_path=data/train.json --save_path=info/train.txt
"""
