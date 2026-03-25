import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pdfplumber
from gensim.models import Word2Vec
from cbow import CBOWScratch
from skipgram import SkipGramScratch
from sklearn.metrics.pairwise import cosine_similarity

sentences = []

# -------- WORD SIMILARITY FUNCTIONS --------
def word_similarity_gensim(model, word1, word2):
    if word1 not in model.wv or word2 not in model.wv:
        return None
    v1 = model.wv[word1].reshape(1,-1)
    v2 = model.wv[word2].reshape(1,-1)
    return cosine_similarity(v1, v2)[0][0]

def word_similarity_scratch(model, embeddings, word1, word2):
    if word1 not in model.word2idx or word2 not in model.word2idx:
        return None
    v1 = embeddings[model.word2idx[word1]].reshape(1,-1)
    v2 = embeddings[model.word2idx[word2]].reshape(1,-1)
    return cosine_similarity(v1, v2)[0][0]

# -------- WORD PAIRS --------
test_pairs = [
    ("course","semester"),
    ("student","research"),
    ("student","register"),
]

# -------- LOAD CORPUS --------
with open("cleaned_corpus.txt", "r") as f:
    for line in f:
        words = line.strip().split()
        if words:
            sentences.append(words)

# -------- HYPERPARAMETERS --------
embedding_dimensions = [100,200]
window_sizes = [5,7]
negative_samples = [5,7]

# -------- RESULT STORAGE --------
results_cbow = {}
results_skipgram = {}

# =====================================================
# CBOW SCRATCH
# =====================================================
print("\nTraining CBOW From Scratch\n")
for dim in embedding_dimensions:
    for window in window_sizes:
        label = f"CBOW_Scratch_dim{dim}_win{window}"
        print(label)
        model = CBOWScratch(sentences, dim=dim, window=window, epochs=5)
        embeddings = model.train()

        scores = []
        print("\nWord Similarity Evaluation")
        for w1, w2 in test_pairs:
            score = word_similarity_scratch(model, embeddings, w1, w2)
            if score is not None:
                print(f"{w1} - {w2} : {score:.4f}")
                scores.append(score)
            else:
                scores.append(0)
        results_cbow[label] = scores
        np.save(f"{label}.npy", embeddings)

# =====================================================
# CBOW GENSIM
# =====================================================
print("\nTraining CBOW Models (Gensim)\n")
for dim in embedding_dimensions:
    for window in window_sizes:
        for neg in negative_samples:
            label = f"CBOW_Gensim_dim{dim}_win{window}_neg{neg}"
            print(label)
            model = Word2Vec(
                sentences=sentences,
                vector_size=dim,
                window=window,
                negative=neg,
                sg=0,
                min_count=2,
                workers=4,
                epochs=10
            )
            scores = []
            print("\nWord Similarity Evaluation")
            for w1, w2 in test_pairs:
                score = word_similarity_gensim(model, w1, w2)
                if score is not None:
                    print(f"{w1} - {w2} : {score:.4f}")
                    scores.append(score)
                else:
                    scores.append(0)
            results_cbow[label] = scores
            model.save(f"{label}.model")

# =====================================================
# SKIPGRAM SCRATCH
# =====================================================
print("\nTraining Skip-gram From Scratch\n")
for dim in embedding_dimensions:
    for window in window_sizes:
        label = f"SkipGram_Scratch_dim{dim}_win{window}"
        print(label)
        model = SkipGramScratch(sentences, dim=dim, window=window, epochs=5)
        embeddings = model.train()
        scores = []
        print("\nWord Similarity Evaluation")
        for w1, w2 in test_pairs:
            score = word_similarity_scratch(model, embeddings, w1, w2)
            if score is not None:
                print(f"{w1} - {w2} : {score:.4f}")
                scores.append(score)
            else:
                scores.append(0)
        results_skipgram[label] = scores
        np.save(f"{label}.npy", embeddings)

# =====================================================
# SKIPGRAM GENSIM
# =====================================================
print("\nTraining Skip-gram Models (Gensim)\n")
for dim in embedding_dimensions:
    for window in window_sizes:
        for neg in negative_samples:
            label = f"SkipGram_Gensim_dim{dim}_win{window}_neg{neg}"
            print(label)
            model = Word2Vec(
                sentences=sentences,
                vector_size=dim,
                window=window,
                negative=neg,
                sg=1,
                min_count=2,
                workers=4,
                epochs=10
            )
            scores = []
            print("\nWord Similarity Evaluation")
            for w1, w2 in test_pairs:
                score = word_similarity_gensim(model, w1, w2)
                if score is not None:
                    print(f"{w1} - {w2} : {score:.4f}")
                    scores.append(score)
                else:
                    scores.append(0)
            results_skipgram[label] = scores
            model.save(f"{label}.model")

print("\nTraining completed.")

# =====================================================
# PLOT CBOW COMPARISON
# =====================================================
labels_pairs = [f"{w1}-{w2}" for w1, w2 in test_pairs]

plt.figure(figsize=(12,6))
for model_name, scores in results_cbow.items():
    plt.plot(labels_pairs, scores, marker='o', label=model_name)
plt.title("CBOW: Scratch vs Gensim")
plt.xlabel("Word Pairs")
plt.ylabel("Cosine Similarity")
plt.xticks(rotation=15)
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================================================
# PLOT SKIPGRAM COMPARISON
# =====================================================
plt.figure(figsize=(12,6))
for model_name, scores in results_skipgram.items():
    plt.plot(labels_pairs, scores, marker='o', label=model_name)
plt.title("Skip-gram: Scratch vs Gensim")
plt.xlabel("Word Pairs")
plt.ylabel("Cosine Similarity")
plt.xticks(rotation=15)
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()