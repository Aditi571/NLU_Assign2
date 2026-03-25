import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD TRAINING DATA ----------------
with open("TrainingNames.txt") as f:
    names = [line.strip().lower() for line in f.readlines()]

chars = sorted(list(set("".join(names))))
chars = ['<s>', '<e>'] + chars

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

training_set = set(names)
vocab_size = len(chars)

# ---------------- MODEL CLASSES ----------------
embedding_dim = 32
hidden_size = 128
num_layers = 1

class VanillaRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        out,_ = self.rnn(x)
        out = self.fc(out)
        return out

class BLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, vocab_size)
    def forward(self,x):
        x = self.embedding(x)
        out,_ = self.lstm(x)
        out = self.fc(out)
        return out

class AttentionRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Linear(hidden_size,1,bias=False)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self,x):
        x = self.embedding(x)
        rnn_out,_ = self.rnn(x)
        score = torch.tanh(self.attention(rnn_out))
        weights = torch.softmax(self.context(score),dim=1)
        context = (weights * rnn_out).sum(dim=1)
        context = context.unsqueeze(1).repeat(1,rnn_out.size(1),1)
        out = self.fc(context)
        return out

# ---------------- GENERATE NAMES ----------------
def generate_name(model, max_len=20):
    model.eval()
    with torch.no_grad():
        char = torch.tensor([[stoi['<s>']]]).to(device)
        name = ""
        for _ in range(max_len):
            output = model(char)
            probs = torch.softmax(output[0,-1], dim=0)
            idx = torch.multinomial(probs, 1).item()
            if itos[idx] == '<e>':
                break
            name += itos[idx]
            char = torch.cat([char, torch.tensor([[idx]]).to(device)], dim=1)
    return name

def generate_names(model, n=500):
    return [generate_name(model) for _ in range(n)]

def novelty_rate(generated, training_set):
    return sum(1 for name in generated if name not in training_set) / len(generated)

def diversity(generated):
    return len(set(generated)) / len(generated)

# ---------------- LOAD MODELS ----------------
models = {
    "Vanilla RNN": ("vanilla_rnn.pth", VanillaRNN),
    "BLSTM": ("blstm.pth", BLSTM),
    "RNN + Attention": ("attention_rnn.pth", AttentionRNN)
}

# ---------------- EVALUATE ----------------
for name, (path, ModelClass) in models.items():
    model = ModelClass().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    generated = generate_names(model, 500)
    novelty = novelty_rate(generated, training_set) * 100
    div = diversity(generated) * 100
    print(f"\nModel: {name}")
    print(f"Novelty Rate: {novelty:.2f}%")
    print(f"Diversity: {div:.2f}%")