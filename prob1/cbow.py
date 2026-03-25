import numpy as np
from collections import Counter

class CBOWScratch:

    def __init__(self, sentences, dim=100, window=5, lr=0.05, epochs=5):
        self.sentences = sentences
        self.dim = dim
        self.window = window
        self.lr = lr
        self.epochs = epochs

        words = [w for s in sentences for w in s]
        vocab = list(set(words))

        self.word2idx = {w:i for i,w in enumerate(vocab)}
        self.idx2word = {i:w for w,i in self.word2idx.items()}
        self.V = len(vocab)

        self.W1 = np.random.randn(self.V, dim) * 0.01
        self.W2 = np.random.randn(dim, self.V) * 0.01

    def train(self):

        for _ in range(self.epochs):

            for sent in self.sentences:

                for i in range(self.window, len(sent)-self.window):

                    context = sent[i-self.window:i] + sent[i+1:i+self.window+1]
                    target = sent[i]

                    context_ids = [self.word2idx[w] for w in context if w in self.word2idx]
                    target_id = self.word2idx[target]

                    h = np.mean(self.W1[context_ids], axis=0)

                    u = np.dot(h, self.W2)
                    y_pred = np.exp(u) / np.sum(np.exp(u))

                    e = y_pred
                    e[target_id] -= 1

                    dW2 = np.outer(h, e)
                    dW1 = np.dot(self.W2, e)

                    self.W2 -= self.lr * dW2

                    for idx in context_ids:
                        self.W1[idx] -= self.lr * dW1 / len(context_ids)

        return self.W1