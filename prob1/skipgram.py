import numpy as np

class SkipGramScratch:

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

                for i, word in enumerate(sent):

                    if word not in self.word2idx:
                        continue

                    w_id = self.word2idx[word]

                    start = max(0, i-self.window)
                    end = min(len(sent), i+self.window+1)

                    context = sent[start:i] + sent[i+1:end]

                    h = self.W1[w_id]

                    for ctx in context:

                        if ctx not in self.word2idx:
                            continue

                        ctx_id = self.word2idx[ctx]

                        u = np.dot(h, self.W2)
                        y_pred = np.exp(u) / np.sum(np.exp(u))

                        e = y_pred
                        e[ctx_id] -= 1

                        dW2 = np.outer(h, e)
                        dW1 = np.dot(self.W2, e)

                        self.W2 -= self.lr * dW2
                        self.W1[w_id] -= self.lr * dW1

        return self.W1