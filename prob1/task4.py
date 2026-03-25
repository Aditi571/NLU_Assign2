from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# load trained models
cbow_model = Word2Vec.load("CBOW_Gensim_dim200_win7_neg7.model")
skipgram_model = Word2Vec.load("SkipGram_Gensim_dim200_win7_neg7.model")

# words related to academics
words = [
    "research","student","programme","regular","course",
    "grade","department","tech","degree","semester",
    "examination","dean","thesis","engineering"
]
def visualize(model, title, method="pca"):
    vectors = []
    labels = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
            labels.append(word)
    vectors = np.array(vectors)

    if method == "pca":
        reducer = PCA(n_components=2)
    else:
        perplexity = min(5, len(vectors) - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)

    reduced_vectors = reducer.fit_transform(vectors)

    plt.figure(figsize=(8,6))

    for i, label in enumerate(labels):
        x = reduced_vectors[i][0]
        y = reduced_vectors[i][1]

        plt.scatter(x, y)
        plt.text(x+0.01, y+0.01, label)

    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()


# visualize CBOW
visualize(cbow_model, "CBOW Word Embedding Visualization (PCA)", method="pca")

# visualize Skip-gram
visualize(skipgram_model, "Skip-gram Word Embedding Visualization (PCA)", method="pca")

# visualize Skip-gram with t-SNE
visualize(skipgram_model, "Skip-gram Word Embedding Visualization (t-SNE)", method="tsne")