#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to process data.
"""
import os
import pickle
import logging
from collections import Counter

import numpy as np
from util import read_conll, one_hot, window_iterator, ConfusionMatrix, load_word_vector_mapping
from defs import LBLS, NONE, LMAP, NUM, UNK, EMBED_SIZE

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


FDIM = 4
P_CASE = "CASE:"
CASES = ["aa", "AA", "Aa", "aA"]
START_TOKEN = "<s>"
END_TOKEN = "</s>"

def casing(word):
    '''词的大小写属性'''
    if len(word) == 0: return word

    # all lowercase
    if word.islower(): return "aa"
    # all uppercase
    elif word.isupper(): return "AA"
    # starts with capital
    elif word[0].isupper(): return "Aa"
    # has non-initial capital
    else: return "aA"

def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    if word.isdigit(): return NUM
    else: return word.lower()

def featurize(embeddings, word):
    """
    Featurize a word given embeddings.
    """
    case = casing(word)
    word = normalize(word)
    case_mapping = {c: one_hot(FDIM, i) for i, c in enumerate(CASES)}
    wv = embeddings.get(word, embeddings[UNK])
    fv = case_mapping[case]
    return np.hstack((wv, fv))

def evaluate(model, X, Y):
    cm = ConfusionMatrix(labels=LBLS)
    Y_ = model.predict(X)
    for i in range(Y.shape[0]):
        y, y_ = np.argmax(Y[i]), np.argmax(Y_[i])
        cm.update(y,y_)
    cm.print_table()
    return cm.summary()

class ModelHelper(object):
    """
    This helper takes care of preprocessing data, constructing embeddings, etc.
    """
    def __init__(self, tok2id, max_length):
        self.tok2id = tok2id # 标记-索引表
        self.START = [tok2id[START_TOKEN], tok2id[P_CASE + "aa"]]
        self.END = [tok2id[END_TOKEN], tok2id[P_CASE + "aa"]]
        self.max_length = max_length

    def vectorize_example(self, sentence, labels=None):
        # 以词的索引和词的大小写属性作为词的特征,表征一个词,sentences=[[f1,f2]_w1,...,[f1,f2]_wn]
        sentence_ = [[self.tok2id.get(normalize(word), self.tok2id[UNK]), self.tok2id[P_CASE + casing(word)]] for word in sentence]
        if labels:
            labels_ = [LBLS.index(l) for l in labels] #句子中每个词对应的命名实体的索引
            return sentence_, labels_
        else: 
            # 所有的词的实体设置为O,缺省类,返回([sentences], [labels])
            return sentence_, [LBLS[-1] for _ in sentence]

    def vectorize(self, data):
        # [([sentences1], [labels1]), ...]
        return [self.vectorize_example(sentence, labels) for sentence, labels in data]

    @classmethod # 声明为类方法, cls指向ModelHelper类,即类本身,但不是实例,实例为self
    def build(cls, data):
        # Preprocess data to construct an embedding
        # Reserve 0 for the special NIL token. 构建标识-索引表
        # data中所有的词转换为小写,按词频排序
        # 返回:{'sloga': 40, ... ,'suicidal': 1071}
        tok2id = build_dict((normalize(word) for sentence, _ in data for word in sentence), offset=1, max_words=10000)
        # 将大小写类型标识追加到tok2id表的后面
        tok2id.update(build_dict([P_CASE + c for c in CASES], offset=len(tok2id)))
        # 将start,end,unk标识追加到tok2id表的后面
        tok2id.update(build_dict([START_TOKEN, END_TOKEN, UNK], offset=len(tok2id)))
        # tok2id 的索引从1开始
        assert sorted(tok2id.items(), key=lambda t: t[1])[0][1] == 1
        logger.info("Built dictionary for %d features.", len(tok2id))

        # data中最长句子的大小
        max_length = max(len(sentence) for sentence, _ in data)

        return cls(tok2id, max_length) # 返回一个当前类的实例

    def save(self, path):
        # Make sure the directory exists.
        if not os.path.exists(path):
            os.makedirs(path)
        # Save the tok2id map.
        with open(os.path.join(path, "features.pkl"), "w") as f:
            pickle.dump([self.tok2id, self.max_length], f) # 以helper类保存tok2id表

    @classmethod
    def load(cls, path):
        # Make sure the directory exists.
        assert os.path.exists(path) and os.path.exists(os.path.join(path, "features.pkl"))
        # Save the tok2id map.
        with open(os.path.join(path, "features.pkl")) as f:
            tok2id, max_length = pickle.load(f)
        return cls(tok2id, max_length)

def load_and_preprocess_data(args):
    logger.info("Loading training data...")
    train = read_conll(args.data_train)
    logger.info("Done. Read %d sentences", len(train))
    logger.info("Loading dev data...")
    dev = read_conll(args.data_dev)
    logger.info("Done. Read %d sentences", len(dev))

    helper = ModelHelper.build(train)
    
    # now process all the input data.
    train_data = helper.vectorize(train)
    dev_data = helper.vectorize(dev)

    return helper, train_data, dev_data, train, dev

def load_embeddings(args, helper):
    embeddings = np.array(np.random.randn(len(helper.tok2id) + 1, EMBED_SIZE), dtype=np.float32)
    embeddings[0] = 0.
    # vocab:词汇表, vectors:对应的向量表,将词汇转换为对应的向量
    # 返回: {word:[word vector element],...}
    for word, vec in load_word_vector_mapping(args.vocab, args.vectors).items():
        word = normalize(word)
        if word in helper.tok2id:
            # 从包含所有词向量的字典中生成本数据集的编码矩阵
            embeddings[helper.tok2id[word]] = vec
    logger.info("Initialized embeddings.")

    return embeddings

def build_dict(words, max_words=None, offset=0):
    cnt = Counter(words)
    if max_words:
        words = cnt.most_common(max_words) # 词频有序
    else:
        words = cnt.most_common()
    return {word: offset+i for i, (word, _) in enumerate(words)}


def get_chunks(seq, default=LBLS.index(NONE)):
    """Breaks input of 4 4 4 0 0 4 0 ->   (0, 4, 5), (0, 6, 7)"""
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1, tok 为 O,Other
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None
        # End of a chunk + start of a chunk!
        elif tok != default:
            if chunk_type is None:
                chunk_type, chunk_start = tok, i
            elif tok != chunk_type:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks

def test_get_chunks():
    assert get_chunks([4, 4, 4, 0, 0, 4, 1, 2, 4, 3], 4) == [(0,3,5), (1, 6, 7), (2, 7, 8), (3,9,10)]
