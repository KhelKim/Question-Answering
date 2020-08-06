from utils import read_text, read_json
from ko_tokenizers import make_texts, preprocess_text
from tqdm import tqdm
import numpy as np
from konlpy.tag import Mecab
from gensim.models import Word2Vec

data_path = "info/train.txt"
vocab_path = "info/mecab-vocab.json"
save_path = "info/mecab-embedding"
embedding_size = 100
vocab_size = 10000
seed = 100

np.random.seed(seed=seed)
tokenizer = Mecab()

data = read_text(data_path)
data = make_texts(data)
data = [preprocess_text(text) for text in data]

tokenized_texts = [tokenizer.morphs(text) for text in tqdm(data)]

model = Word2Vec(sentences=tokenized_texts, size=embedding_size, window=5,
                 min_count=5, workers=4, sg=0)

word2vec = model.wv
vocab = read_json(vocab_path)

vectors = []
for key, value in vocab.items():
    try:
        vectors.append(word2vec[key])
    except KeyError:
        random = np.random.normal(size=embedding_size)
        vectors.append(random)

word_embedding = np.array(vectors)[:vocab_size]
np.save(save_path, word_embedding)
