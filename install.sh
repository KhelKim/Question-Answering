touch dataloader.py convert_json2txt.py evaluation.py
touch ko_tokenizers.py main.py readme.md trainer.py utils.py
mkdir data info models saved_model
mkdir data/test data/train data/validate
mkdir models/bidaf_imple models/doc_qa
unzip /datasets/objstrgzip/15_MRC_Q\&A.zip
mv test.json data
mv train.json data
mv validate.json data
rm vocab.json


pip3 install transformers==2.11.0 wandb tokenizers==0.7.0 tqdm==4.45.0
pip3 install beautifulsoup4==4.9.1 lxml==4.5.1
pip3 install konlpy==0.5.2
pip3 install hgtk==0.1.0
pip3 install mecab-python3==0.996.5
pip3 install gensim==3.8.3
