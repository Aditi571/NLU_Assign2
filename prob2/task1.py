import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA PREPARATION
with open("TrainingNames.txt") as f:
    names = [line.strip().lower() for line in f.readlines()]

chars = sorted(list(set("".join(names))))
chars = ['<s>', '<e>'] + chars

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i, ch in stoi.items()}

vocab_size = len(chars)

def encode(name):
    return [stoi['<s>']] + [stoi[c] for c in name] + [stoi['<e>']]

encoded_names = [encode(n) for n in names]

# HYPERPARAMETERS
embedding_dim = 32
hidden_size = 128
num_layers = 1
learning_rate = 0.003
epochs = 10

# PARAMETER COUNT FUNCTION
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# MODEL 1 : VANILLA RNN
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

# MODEL 2 : BIDIRECTIONAL LSTM
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

# MODEL 3 : RNN + ATTENTION
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

# TRAINING FUNCTION WITH LOSS TRACKING
def train_model(model):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        for name in encoded_names:
            x = torch.tensor(name[:-1]).unsqueeze(0).to(device)
            y = torch.tensor(name[1:]).unsqueeze(0).to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out.view(-1,vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(encoded_names)
        loss_history.append(avg_loss)
        print(f"Epoch: {epoch+1}, Loss: {avg_loss:.4f}")

    return loss_history

# TRAIN AND PLOT
models = {
    "Vanilla RNN": VanillaRNN(),
    "BLSTM": BLSTM(),
    "Attention RNN": AttentionRNN()
}

plt.figure(figsize=(10,6))

for name, model in models.items():
    print(f"\nTraining {name}")
    print("Parameters:", count_parameters(model))
    losses = train_model(model)
    plt.plot(range(1, epochs+1), losses, label=name)

    # Save models
    torch.save(model.state_dict(), f"{name.replace(' ','_').lower()}.pth")

plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()

print("Models trained and saved successfully!")