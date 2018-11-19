# _*_ coding:utf8 _*_

import numpy as np 
import matplotlib.pyplot as plt

def svd_demo():
    # svd分解窗口共现矩阵X,并降维分析词共现之间的关系
    # 词库
    words = ["I", "like", "enjoy", "deep", "learnig", "NLP", "flying", "."]
    # 共现矩阵
    X = np.array([[0,2,1,0,0,0,0,0],
                  [2,0,0,1,0,1,0,0],
                  [1,0,0,0,0,0,1,0],
                  [0,1,0,0,1,0,0,0],
                  [0,0,0,1,0,0,0,1],
                  [0,1,0,0,0,0,0,1],
                  [0,0,1,0,0,0,0,1],
                  [0,0,0,0,1,1,1,0]])

    U, s, V = np.linalg.svd(X, full_matrices=False)
    plt.axis([-1, 1, -1, 1])
    for i in xrange(len(words)):
        plt.text(U[i,0], U[i,1], words[i])
    plt.show()
    
svd_demo()
