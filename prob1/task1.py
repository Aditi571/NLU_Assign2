import os
import re
import pdfplumber
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

data_folder = "../data"

documents = []
all_tokens = []

for file in os.listdir(data_folder):

    if file.endswith(".pdf"):
        path = os.path.join(data_folder, file)

        text = ""

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "

        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        tokens = text.split()
        tokens = [word for word in tokens if len(word) > 2]
        documents.append(tokens)
        all_tokens.extend(tokens)


# ----------saving cleaned corpus----------

with open("cleaned_corpus.txt", "w") as f:
    for doc in documents:
        f.write(" ".join(doc))
        f.write("\n")

print("Cleaned corpus saved to cleaned_corpus.txt")


# ----------Dataset statistics ----------

total_documents = len(documents)
total_tokens = len(all_tokens)
vocab = set(all_tokens)
vocab_size = len(vocab)
print("\nDataset Statistics")
print("\n")
print("Total Documents are:", total_documents)
print("Total Tokens are:", total_tokens)
print("Vocabulary Size : ", vocab_size)


word_freq = Counter(all_tokens)

print("\nTop 10 most frequent words:")
for word, freq in word_freq.most_common(10):
    print(word, freq)

# ----------Word cloud formation ----------

if len(all_tokens) == 0:
    print("No words found in PDFs")
else:
    text_for_cloud = " ".join(all_tokens)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text_for_cloud)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of IIT Jodhpur Text Data")
    plt.show()