from gensim.models import Word2Vec

# Load your trained Word2Vec model
model = Word2Vec.load("SkipGram_Gensim_dim200_win7_neg5.model")  # replace with your model filename

word = "research"

if word in model.wv:
    vector = model.wv[word]
    vector_str = ", ".join([f"{v:.4f}" for v in vector])
    print(f"{word} - {vector_str}")
else:
    print(f"Word '{word}' not found in the vocabulary.")