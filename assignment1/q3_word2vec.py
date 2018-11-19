#!/usr/bin/env python
# _*_ coding:utf8 _*_

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    将矩阵每一行的元素除以该行元素平方和的开平方,
    \frac{x_i}{\sqrt{\sum_{i=1}^d x_i^2}},x为行向量
    """

    ### YOUR CODE HERE
    # denom = np.apply_along_axis(lambda x: np.sqrt(x.T.dot(x)), 1, x)
    denom = np.apply_along_axis(lambda x: np.sqrt(np.dot(x, x.T)), 1, x)
    # x /= denom[:, None]
    x /= denom
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    print x
    ans = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word,期望得到的单词下标
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    ## Gradient for $\hat{\bm{v}}$:

    #  Calculate the predictions:
    # 中心词向量
    vhat = predicted 
    # 计算U x v_c, U={u_0,u_1,u_2,u_3,u_4}
    # 对应词汇{a,b,c,d,e}的上下文向量,U是5x3矩阵
    # 5为独立词个数,即词库大小,3为词向量维度,[5x3]x[3x1] = 5x1
    z = np.dot(outputVectors, vhat)
    # 得到当前中心词下,对应每一个上下文词汇的预测概率值
    preds = softmax(z) # 5x1

    #  Calculate the cost:
    # 计算期望得到的输出上下文词向量的损失值
    cost = -np.log(preds[target])

    # Gradients 
    # y^hat - y = [yh_1,yh_2,yh_3,yh_4,yh_5] - [0,0,1,0,0],y_3是ground_truth标签
    # y^hat 为已经得到的在当前中心词下,每个词汇的预测概率
    # y为one-hat编码,指示在当前中心词下,期望获得的上下文词向量
    z = preds.copy()
    # [yh_1, yh_2, yh_3 - 1, yh_4, yh_5],yh_3为期望得到的输出词汇
    z[target] -= 1.0

    # 计算U的梯度, \frac{\partial J}{\partial U} = v_c * (y^hat - y), [5x3]
    grad = np.outer(z, vhat)
    # 计算v_c的梯度, grad_v = \frac{\partial J}{partial v_c} = -u_o + \sum_i^w yh_i*u_i
    # grad_v = -u_o + yh_1*u_1 + ... + yh_o*u_o + ... + yh_w*u_w
    # grad_v = yh_1*u_1 + ... + (yh_o - 1)*u_o + ... + yh_w*u_w
    # [3x5]x[5x1] = [3x1]
    gradPred = np.dot(outputVectors.T, z) 
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            # 与期望得到的词汇索引一致,则重新采样负样本
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    # WxD,W为词库大小
    grad = np.zeros(outputVectors.shape)
    # 1xD,D为词向量的维度
    gradPred = np.zeros(predicted.shape)
    cost = 0
    z = sigmoid(np.dot(outputVectors[target], predicted))

    cost -= np.log(z)
    grad[target] += predicted * (z - 1.0)
    gradPred += outputVectors[target] * (z - 1.0)

    for k in xrange(K):
        samp = indices[k + 1]
        # sigmoid(u_k*v_c)
        z = sigmoid(np.dot(outputVectors[samp], predicted))
        cost -= np.log(1.0 - z)
        # (1 - sigmoid(-u_k*v_c)) = sigmoid(u_k*v_c)
        grad[samp] += predicted * z
        gradPred += outputVectors[samp] * z
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word 当前的中心词汇
    C -- integer, context size 上下文窗口大小
    contextWords -- list of no more than 2*C strings, the context words 窗口大小的文本
    tokens -- a dictionary that maps words to their indices in
              the word vector list 指示词汇在词库中位置的索引表
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape) # WxD,中心词向量梯度矩阵,包含词库中所有的词,初始化为0
    gradOut = np.zeros(outputVectors.shape) # WxD,上下文词向量梯度矩阵,包含词库中所有的词,初始化为0

    ### YOUR CODE HERE
    # 获取中心词向量
    cword_idx = tokens[currentWord]
    vhat = inputVectors[cword_idx]

    for j in contextWords:
        # 计算在当前中心词向量下,该窗口内的每个词的上下文词向量对应的代价和梯度
        u_idx = tokens[j]
        c_cost, c_grad_in, c_grad_out = \
            word2vecCostAndGradient(vhat, u_idx, outputVectors, dataset)
        # 累加该中心词下窗口内的所有上下文词汇的代价,梯度
        cost += c_cost
        # 累加中心词自身的梯度
        gradIn[cword_idx] += c_grad_in
        gradOut += c_grad_out
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    skim-gram模型和CBOW(连续词袋模型)的目的是一样的,就是要求的中心词和上下文词汇
    的向量表示,CBOW从上下文词汇本身入手,去估计中心词向量,然后通过估计值和真实值的
    代价函数,训练模型,求解出中心词向量和上下文词向量.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    predicted_indices = [tokens[word] for word in contextWords]
    predicted_vectors = inputVectors[predicted_indices]
    # 用上下文词汇来估计中心词汇
    predicted = np.sum(predicted_vectors, axis=0)
    # 中心词汇作为期望得到的输出词汇,训练模型
    target = tokens[currentWord]
    cost, gradIn_predicted, gradOut = word2vecCostAndGradient(predicted, target, outputVectors, dataset)
    for i in predicted_indices:
        gradIn[i] += gradIn_predicted
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0] # N/2为词库大小
    # 前N个为中心词数量,后N个为上下文词汇数量
    inputVectors = wordVectors[:N / 2, :]
    outputVectors = wordVectors[N / 2:, :]
    for i in xrange(batchsize):
        # 一个窗口就是一条训练数据
        C1 = random.randint(1, C)
        # 随机取出1个中心词和中心词左右窗口大小的上下文词汇
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        # 代价和梯度按训练批次数量取平均值
        cost += c / batchsize / denom
        grad[:N / 2, :] += gin / batchsize / denom
        grad[N / 2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    # type创建一个空类,并生成实例
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        # 从[0,4)间随机采样一个值
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0, 4)], \
               [tokens[random.randint(0, 4)] for i in xrange(2 * C)]

    # 为空类定义两个函数操作,并使用这个内部类来传函数
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    # 初始化构建词向量,前5行为中心词向量,后5行为上下文词向量
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
                    dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
    print skipgram("c", 1, ["a", "b"],
                   dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
                   negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
               dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset,
               negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
