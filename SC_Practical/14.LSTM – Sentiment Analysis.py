import torch
import torch.nn as nn
import torch.optim as optim

sentences = [
    "I love this movie",
    "This film is great",
    "Amazing experience",
    "I hate this movie",
    "This film is terrible",
    "Worst experience"
]
labels = [1, 1, 1, 0, 0, 0]   

vocab = {}
for s in sentences:
    for w in s.lower().split():
        if w not in vocab:
            vocab[w] = len(vocab) + 1

def encode(sentence):
    return [vocab.get(w, 0) for w in sentence.lower().split()]

X = [encode(s) for s in sentences]
y = torch.tensor(labels, dtype=torch.float)


max_len = max(len(seq) for seq in X)
X_pad = [seq + [0]*(max_len - len(seq)) for seq in X]
X_tensor = torch.tensor(X_pad)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, 8, padding_idx=0)
        self.lstm = nn.LSTM(8, 16, batch_first=True)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        return self.sigmoid(self.fc(h[-1]))

model = LSTMClassifier(len(vocab))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


for epoch in range(200):
    optimizer.zero_grad()
    output = model(X_tensor).squeeze()
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()


test_sentence = "I love this film"
test_encoded = encode(test_sentence)
test_encoded += [0] * (max_len - len(test_encoded))
test_tensor = torch.tensor([test_encoded])

prediction = model(test_tensor).item()
print("Sentiment score (0=negative, 1=positive):", prediction)