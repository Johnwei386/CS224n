# _*_ coding:utf8 _*_

"""Utilities for training the dependency parser.
You do not need to read/understand this code
"""

import time
import os
import logging
from collections import Counter
from general_utils import logged_loop, get_minibatches
from q2_parser_transitions import PartialParse, minibatch_parse

import numpy as np


P_PREFIX = '<p>:'
L_PREFIX = '<l>:'
UNK = '<UNK>'
NULL = '<NULL>'
ROOT = '<ROOT>'


class Config(object):
    language = 'english'
    with_punct = True
    unlabeled = True
    lowercase = True
    use_pos = True
    use_dep = True
    use_dep = use_dep and (not unlabeled) # False
    data_path = './data'
    train_file = 'train.conll'
    dev_file = 'dev.conll'
    test_file = 'test.conll'
    embedding_file = './data/en-cw.txt'


class Parser(object):
    """Contains everything needed for transition-based dependency parsing except for the model"""

    def __init__(self, dataset):
        # 得到句子的根依赖
        root_labels = list([l for ex in dataset
                           for (h, l) in zip(ex['head'], ex['label']) if h == 0])
        # {'root': 1000}
        counter = Counter(root_labels)
        
        if len(counter) > 1:
            logging.info('Warning: more than one root label')
            logging.info(counter)
        
        # [('root', 1000)]
        self.root_label = counter.most_common()[0][0]
        # 所有的独立标签项
        deprel = [self.root_label] + list(set([w for ex in dataset
                                               for w in ex['label']
                                               if w != self.root_label]))
        
        # 标签项
        # '<l>:root': 0, '<l>:nmod': 32
        tok2id = {L_PREFIX + l: i for (i, l) in enumerate(deprel)}
        # 新加了一项,现在是39项, '<l>:<NULL>': 38, 
        tok2id[L_PREFIX + NULL] = self.L_NULL = len(tok2id)

        config = Config()
        self.unlabeled = config.unlabeled 
        self.with_punct = config.with_punct
        self.use_pos = config.use_pos
        self.use_dep = config.use_dep
        self.language = config.language

        if self.unlabeled:
            # 定义标签无关的操作集合
            trans = ['L', 'R', 'S']
            # 依赖类型数目,标记无关就只有1种
            self.n_deprel = 1
        else:
            # 定义标签相关的操作集合,38+38+1
            trans = ['L-' + l for l in deprel] + ['R-' + l for l in deprel] + ['S']
            self.n_deprel = len(deprel)

        self.n_trans = len(trans) # 操作集大小
        self.tran2id = {t: i for (i, t) in enumerate(trans)} # 操作-索引
        self.id2tran = {i: t for (i, t) in enumerate(trans)} # 索引-操作

        # 加入词法项
        # logging.info('Build dictionary for part-of-speech tags.')
        # ex指向句子,w指向句子中的每条依赖的词法项
        tok2id.update(build_dict([P_PREFIX + w for ex in dataset for w in ex['pos']],
                                  offset=len(tok2id)))
        tok2id[P_PREFIX + UNK] = self.P_UNK = len(tok2id)
        tok2id[P_PREFIX + NULL] = self.P_NULL = len(tok2id)
        tok2id[P_PREFIX + ROOT] = self.P_ROOT = len(tok2id)

        # 加入单词项(依赖的尾端)
        # logging.info('Build dictionary for words.')
        tok2id.update(build_dict([w for ex in dataset for w in ex['word']],
                                  offset=len(tok2id)))
        tok2id[UNK] = self.UNK = len(tok2id)
        tok2id[NULL] = self.NULL = len(tok2id)
        tok2id[ROOT] = self.ROOT = len(tok2id)

        self.tok2id = tok2id
        self.id2tok = {v: k for (k, v) in tok2id.items()} 

        # 特征维度
        self.n_features = 18 + (18 if config.use_pos else 0) + (12 if config.use_dep else 0)
        self.n_tokens = len(tok2id) # 总的标记项,5157

    def vectorize(self, examples):
        vec_examples = []
        # 一次处理一个句子
        for ex in examples:
            # 得到句子中的单词项,在tokens中的索引
            word = [self.ROOT] + [self.tok2id[w] if w in self.tok2id
                                  else self.UNK for w in ex['word']]
            # 得到句子中的词性项,得到索引
            pos = [self.P_ROOT] + [self.tok2id[P_PREFIX + w] if P_PREFIX + w in self.tok2id
                                   else self.P_UNK for w in ex['pos']]
            # 得到依赖项的头,本身就是在句子中的序号,索引
            head = [-1] + ex['head']
            # 得到句子中的标签项,得到索引
            label = [-1] + [self.tok2id[L_PREFIX + w] if L_PREFIX + w in self.tok2id
                            else -1 for w in ex['label']]
            vec_examples.append({'word': word, 'pos': pos,
                                 'head': head, 'label': label})
        return vec_examples

    def extract_features(self, stack, buf, arcs, ex):
        if stack[0] == "ROOT":
            stack[0] = 0 # stack.size至少为1

        def get_lc(k): # 以k为头的弧,弧尾集合,序号比k小,正序
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] < k])

        def get_rc(k): # 以k为头的弧,弧尾集合,序号比k大,逆序
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] > k],
                          reverse=True)

        p_features = []
        l_features = []
        # 考虑stack最后3个词,stack[-3:]若不足3个,则只遍历仅有的几个元素
        # len(stack) = 1 ,features.size = 3
        # len(stack) = 2 ,features.size = 3
        # len(stack) = 3 ,features.size = 3
        features = [self.NULL] * (3 - len(stack)) + [ex['word'][x] for x in stack[-3:]]
        # 考虑buf前3个词,(3 - len(buf))若为负数,则等于0
        # len(buf) = 1 ,features.size = 6
        # len(buf) = 2 ,features.size = 6
        # len(buf) = 3 ,features.size = 6
        features += [ex['word'][x] for x in buf[:3]] + [self.NULL] * (3 - len(buf)) 
        if self.use_pos:
            # p_features.size = 3,考虑stack最后3个词的词性
            p_features = [self.P_NULL] * (3 - len(stack)) + [ex['pos'][x] for x in stack[-3:]]
            # p_features.size = 6,考虑buf前3个词的词性
            p_features += [ex['pos'][x] for x in buf[:3]] + [self.P_NULL] * (3 - len(buf)) # 0为空

        for i in xrange(2): # 0,1,生成剩下的12个特征
            if i < len(stack): # 
                k = stack[-i-1]
                lc = get_lc(k) # 以k为头的序号小于k的词序集合
                rc = get_rc(k) # 以k为头的序号大于k的词序集合
                llc = get_lc(lc[0]) if len(lc) > 0 else [] # 最左边的元素
                rrc = get_rc(rc[0]) if len(rc) > 0 else [] # 最右边的元素

                # 考虑以stack[-1 or -2]为head的弧依赖和其子依赖元素的依赖的影响
                features.append(ex['word'][lc[0]] if len(lc) > 0 else self.NULL)
                features.append(ex['word'][rc[0]] if len(rc) > 0 else self.NULL)
                features.append(ex['word'][lc[1]] if len(lc) > 1 else self.NULL)
                features.append(ex['word'][rc[1]] if len(rc) > 1 else self.NULL)
                features.append(ex['word'][llc[0]] if len(llc) > 0 else self.NULL)
                features.append(ex['word'][rrc[0]] if len(rrc) > 0 else self.NULL)

                if self.use_pos:
                    p_features.append(ex['pos'][lc[0]] if len(lc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][rc[0]] if len(rc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][lc[1]] if len(lc) > 1 else self.P_NULL)
                    p_features.append(ex['pos'][rc[1]] if len(rc) > 1 else self.P_NULL)
                    p_features.append(ex['pos'][llc[0]] if len(llc) > 0 else self.P_NULL)
                    p_features.append(ex['pos'][rrc[0]] if len(rrc) > 0 else self.P_NULL)

                if self.use_dep:
                    l_features.append(ex['label'][lc[0]] if len(lc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][rc[0]] if len(rc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][lc[1]] if len(lc) > 1 else self.L_NULL)
                    l_features.append(ex['label'][rc[1]] if len(rc) > 1 else self.L_NULL)
                    l_features.append(ex['label'][llc[0]] if len(llc) > 0 else self.L_NULL)
                    l_features.append(ex['label'][rrc[0]] if len(rrc) > 0 else self.L_NULL)
            else: # 否则用NULL补齐特征维度, 总共18个特征,l_features是12个特征
                features += [self.NULL] * 6
                if self.use_pos:
                    p_features += [self.P_NULL] * 6
                if self.use_dep:
                    l_features += [self.L_NULL] * 6

        features += p_features + l_features # 18+18+12
        assert len(features) == self.n_features
        return features

    def get_oracle(self, stack, buf, ex):
        if len(stack) < 2:
            # stack中元素少于2,从buf中移动元素过来,执行s
            return self.n_trans - 1

        i0 = stack[-1] # stack top first elem
        i1 = stack[-2] # stack top second elem
        h0 = ex['head'][i0] # h0 -> i0[-1]
        h1 = ex['head'][i1] # h1 -> i1[-2]
        l0 = ex['label'][i0]
        l1 = ex['label'][i1]

        if self.unlabeled:
            if (i1 > 0) and (h1 == i0):
                # s[-1] -> s[-2], left-arc
                return 0
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])): 
                # s[-2] -> s[-1], right-arc,检查buf剩下的项是否是-1的尾
                return 1
            else:
                # shift
                return None if len(buf) == 0 else 2
        else:
            if (i1 > 0) and (h1 == i0):
                return l1 if (l1 >= 0) and (l1 < self.n_deprel) else None
            elif (i1 >= 0) and (h0 == i1) and \
                 (not any([x for x in buf if ex['head'][x] == i0])):
                return l0 + self.n_deprel if (l0 >= 0) and (l0 < self.n_deprel) else None
            else:
                return None if len(buf) == 0 else self.n_trans - 1

    def create_instances(self, examples):
        all_instances = []
        succ = 0
        for id, ex in enumerate(logged_loop(examples)):
            n_words = len(ex['word']) - 1

            # arcs = {(h, t, label)}
            stack = [0]
            # buf初始保存句子中所有的词
            # 这个序号和head集合中的序号是对应的,序号范围在句子的长度内
            buf = [i + 1 for i in xrange(n_words)] 
            arcs = []
            instances = []
            for i in xrange(n_words * 2):
                # 总共会执行2*n_words次trans操作,n为句子中单词总数
                # 得到当前状态下正确的trans操作
                gold_t = self.get_oracle(stack, buf, ex) 
                if gold_t is None:
                    break
                legal_labels = self.legal_labels(stack, buf)
                assert legal_labels[gold_t] == 1
                # 为句子的每一次状态变化创建一个状态向量实例f=[18+18+12]
                # 每个状态向量表征当前未变化前的状态,对应一个正确的可执行操作的标识
                # 总共有2*n_words个状态向量F
                instances.append((self.extract_features(stack, buf, arcs, ex),
                                  legal_labels, gold_t))
                # 更新状态
                if gold_t == self.n_trans - 1: # ==2 shift
                    stack.append(buf[0])
                    buf = buf[1:] 
                elif gold_t < self.n_deprel: # == 0 left-arc
                    arcs.append((stack[-1], stack[-2], gold_t))
                    stack = stack[:-2] + [stack[-1]] # stack去除-2元素
                else: # ==1 right-arc
                    right_label = gold_t if self.unlabeled else gold_t - self.n_deprel
                    arcs.append((stack[-2], stack[-1], right_label)) # 添加新获得的弧,(h,t)对应词序
                    stack = stack[:-1] # stack去除-1元素
            else:
                succ += 1
                all_instances += instances # 将提取回的句子特征加入总特征实例集合

        return all_instances

    def legal_labels(self, stack, buf):
        labels = ([1] if len(stack) > 2 else [0]) * self.n_deprel # l0
        labels += ([1] if len(stack) >= 2 else [0]) * self.n_deprel # l1
        labels += [1] if len(buf) > 0 else [0] # shift
        return labels # 2n+1

    def parse(self, dataset, eval_batch_size=5000):
        sentences = []
        sentence_id_to_idx = {}
        for i, example in enumerate(dataset):
            n_words = len(example['word']) - 1
            sentence = [j + 1 for j in range(n_words)]
            sentences.append(sentence)
            sentence_id_to_idx[id(sentence)] = i

        model = ModelWrapper(self, dataset, sentence_id_to_idx)
        dependencies = minibatch_parse(sentences, model, eval_batch_size)

        UAS = all_tokens = 0.0
        for i, ex in enumerate(dataset):
            head = [-1] * len(ex['word'])
            for h, t, in dependencies[i]:
                head[t] = h
            for pred_h, gold_h, gold_l, pos in \
                    zip(head[1:], ex['head'][1:], ex['label'][1:], ex['pos'][1:]):
                    assert self.id2tok[pos].startswith(P_PREFIX)
                    pos_str = self.id2tok[pos][len(P_PREFIX):]
                    if (self.with_punct) or (not punct(self.language, pos_str)):
                        UAS += 1 if pred_h == gold_h else 0 # 依赖头正确计数
                        all_tokens += 1
        UAS /= all_tokens
        return UAS, dependencies


class ModelWrapper(object):
    def __init__(self, parser, dataset, sentence_id_to_idx):
        self.parser = parser
        self.dataset = dataset
        self.sentence_id_to_idx = sentence_id_to_idx

    def predict(self, partial_parses):
        mb_x = [self.parser.extract_features(p.stack, p.buffer, p.dependencies,
                                             self.dataset[self.sentence_id_to_idx[id(p.sentence)]])
                for p in partial_parses]
        mb_x = np.array(mb_x).astype('int32')
        mb_l = [self.parser.legal_labels(p.stack, p.buffer) for p in partial_parses]
        pred = self.parser.model.predict_on_batch(self.parser.session, mb_x)
        pred = np.argmax(pred + 10000 * np.array(mb_l).astype('float32'), 1)
        pred = ["S" if p == 2 else ("LA" if p == 0 else "RA") for p in pred]
        return pred


def read_conll(in_file, lowercase=False, max_example=None):
    '''从文件中导入数据,返回指定格式的数据集.
       源数据编辑格式: 词序-依赖词-_-_-词性-_-头词-依赖标签-_-_, 每一条为一个依赖.
       句子间以空行分隔,返回一个元组集合,元组里面的每个元素表示一个句子.
    '''
    examples = []
    with open(in_file) as f:
        word, pos, head, label = [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                # 空行大小为0,非空行为句子中的词的依赖解析
                if '-' not in sp[0]:
                    word.append(sp[1].lower() if lowercase else sp[1])
                    pos.append(sp[4])
                    head.append(int(sp[6]))
                    label.append(sp[7])
            elif len(word) > 0:
                # 到达句子的末尾,将这个句子的依赖集合加入数据样本集合中
                examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
                word, pos, head, label = [], [], [], []
                if (max_example is not None) and (len(examples) == max_example):
                    break
        if len(word) > 0:
            examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
    return examples


def build_dict(keys, n_max=None, offset=0):
    count = Counter() # 创建一个空计数器
    for key in keys:
        # 统计某项出现的次数
        count[key] += 1
    # 未指定采样数量,则返回排序后所有项的统计情况
    # [('t1', 100),('t2',98),...]
    ls = count.most_common() if n_max is None \
        else count.most_common(n_max)

    return {w[0]: index + offset for (index, w) in enumerate(ls)}


def punct(language, pos):
    if language == 'english':
        return pos in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
    elif language == 'chinese':
        return pos == 'PU'
    elif language == 'french':
        return pos == 'PUNC'
    elif language == 'german':
        return pos in ["$.", "$,", "$["]
    elif language == 'spanish':
        # http://nlp.stanford.edu/software/spanish-faq.shtml
        return pos in ["f0", "faa", "fat", "fc", "fd", "fe", "fg", "fh",
                       "fia", "fit", "fp", "fpa", "fpt", "fs", "ft",
                       "fx", "fz"]
    elif language == 'universal':
        return pos == 'PUNCT'
    else:
        raise ValueError('language: %s is not supported.' % language)


def minibatches(data, batch_size):
    x = np.array([d[0] for d in data])
    y = np.array([d[2] for d in data])
    one_hot = np.zeros((y.size, 3))
    one_hot[np.arange(y.size), y] = 1
    return get_minibatches([x, one_hot], batch_size)


def load_and_preprocess_data(reduced=True):
    config = Config()

    print "Loading data...",
    start = time.time()
    train_set = read_conll(os.path.join(config.data_path, config.train_file),
                           lowercase=config.lowercase)
    dev_set = read_conll(os.path.join(config.data_path, config.dev_file),
                         lowercase=config.lowercase)
    test_set = read_conll(os.path.join(config.data_path, config.test_file),
                          lowercase=config.lowercase)
    if reduced:
        train_set = train_set[:1000]
        dev_set = dev_set[:500]
        test_set = test_set[:500]
    print "took {:.2f} seconds".format(time.time() - start)

    print "Building parser...",
    # 构建解析器
    start = time.time()
    parser = Parser(train_set)
    print "took {:.2f} seconds".format(time.time() - start)

    print "Loading pretrained embeddings...",
    start = time.time()
    # 得到预编码词向量集合,词向量的维度为50
    word_vectors = {}
    for line in open(config.embedding_file).readlines():
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    # 初始化标记向量矩阵,一个标记可以是词,词性或者依赖的标签,[5157x50]
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (parser.n_tokens, 50)), dtype='float32')

    # 从预编码词向量集合中查找生成对应相应的标记向量
    for token in parser.tok2id:
        i = parser.tok2id[token]
        if token in word_vectors:
            embeddings_matrix[i] = word_vectors[token]
        elif token.lower() in word_vectors:
            embeddings_matrix[i] = word_vectors[token.lower()]
    print "took {:.2f} seconds".format(time.time() - start)

    # 将数据向量化,head为句子中的索引,pos,word和label为tok2id中的索引序号
    # {'head': [-1, 2, 3, 0, 3, 3], 
    #  'word': [5156, 310, 1729, 1284, 4008, 87], 
    #  'pos': [84, 42, 42, 55, 42, 46], 
    #  'label': [-1, 24, 19, 0, 9, 23]}
    # label与head是对应的,-1表示root
    print "Vectorizing data...",
    start = time.time()
    train_set = parser.vectorize(train_set)
    dev_set = parser.vectorize(dev_set)
    test_set = parser.vectorize(test_set)
    print "took {:.2f} seconds".format(time.time() - start)

    print "Preprocessing training data..."
    # 将训练集中的每条数据(每个句子)转化为一组状态向量和正确操作的集合
    train_examples = parser.create_instances(train_set)

    return parser, embeddings_matrix, train_examples, dev_set, test_set,

if __name__ == '__main__':
    pass
