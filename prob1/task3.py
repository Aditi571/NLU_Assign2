import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# ----------loading models ----------

# CBOW Scratch embeddings
cbow_scratch_embeddings = np.load("CBOW_Scratch_dim200_win7.npy", allow_pickle=True)

from cbow import CBOWScratch
with open("cleaned_corpus.txt") as f:
    sentences = [line.strip().split() for line in f if line.strip()]
cbow_scratch_model = CBOWScratch(sentences, dim=200, window=7, epochs=5)

# CBOW Gensim model
cbow_gensim_model = Word2Vec.load("CBOW_Gensim_dim200_win7_neg7.model")

# Skip-gram Scratch embeddings
skipgram_scratch_embeddings = np.load("SkipGram_Scratch_dim200_win7.npy", allow_pickle=True)
from skipgram import SkipGramScratch
skipgram_scratch_model = SkipGramScratch(sentences, dim=200, window=7, epochs=5)

# Skip-gram Gensim model
skipgram_gensim_model = Word2Vec.load("SkipGram_Gensim_dim200_win7_neg7.model")


# ----------WORD SIMILARITY FUNCTIONS FOR SCRATCH MODELS ----------

def word_similarity_scratch(model, embeddings, word1, word2):
    if word1 not in model.word2idx or word2 not in model.word2idx:
        return None
    v1 = embeddings[model.word2idx[word1]].reshape(1,-1)
    v2 = embeddings[model.word2idx[word2]].reshape(1,-1)
    return cosine_similarity(v1, v2)[0][0]


# ----------Nearest Neighbors ----------
words = ["research", "student", "phd", "exam"]

print("\n--- Nearest Neighbors ---\n")

# Example for gensim models
def print_nearest_gensim(model, label):
    print(f"Model: {label}")
    for word in words:
        if word in model.wv:
            print("Word:", word)
            neighbors = model.wv.most_similar(word, topn=5)
            for n, score in neighbors:
                print(f"{n} -> {score:.3f}")
            print()
        else:
            print(word, "not found in vocabulary\n")

print_nearest_gensim(cbow_gensim_model, "CBOW Gensim")
print_nearest_gensim(skipgram_gensim_model, "Skip-gram Gensim")

# For scratch models computing cosine similarity manually
def print_nearest_scratch(model, embeddings, label):
    print(f"Model: {label}")
    for word in words:
        if word in model.word2idx:
            word_vec = embeddings[model.word2idx[word]].reshape(1,-1)
            sims = []
            for w, idx in model.word2idx.items():
                if w == word:
                    continue
                vec = embeddings[idx].reshape(1,-1)
                score = cosine_similarity(word_vec, vec)[0][0]
                sims.append((w, score))
            sims.sort(key=lambda x: x[1], reverse=True)
            print("Word:", word)
            for n, score in sims[:5]:
                print(f"{n} -> {score:.3f}")
            print()
        else:
            print(word, "not found in vocabulary\n")

print_nearest_scratch(cbow_scratch_model, cbow_scratch_embeddings, "CBOW Scratch")
print_nearest_scratch(skipgram_scratch_model, skipgram_scratch_embeddings, "Skip-gram Scratch")


# ----------ANALOGY EXPERIMENTS ----------

# Refined analogies based on your top frequent words
#tech + degree - student
analogies = [
    (["tech", "degree"], ["student"]),
    
    (["academic", "field"], ["regular"]),
    (["degree", "thesis"], ["examination"]),
    
]
def analogy_gensim(model, label):
    print(f"\nAnalogy Results for {label}:\n")
    for positive, negative in analogies:
        try:
            result = model.wv.most_similar(positive=positive, negative=negative, topn=1)
            print(f"{positive} - {negative} -> {result[0][0]} (score: {result[0][1]:.3f})")
        except KeyError:
            print("Some words not found:", positive, negative)

analogy_gensim(cbow_gensim_model, "CBOW Gensim")
analogy_gensim(skipgram_gensim_model, "Skip-gram Gensim")