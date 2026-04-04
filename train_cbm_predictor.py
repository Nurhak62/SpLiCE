import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

# Configuration
train_emb_dir = '/workspaces/SpLiCE/embeddings/splice'
val_emb_dir = '/workspaces/SpLiCE/embeddings/splice_val'
wnid_to_label = {'n02085620': 0, 'n02123159': 1, 'n01443537': 2, 'n01534433': 3, 'n02132136': 4}
num_classes = 5
embedding_dim = 10000
batch_size = 128
num_epochs = 50
learning_rate = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EmbeddingDataset(Dataset):
    def __init__(self, emb_dir, wnid_to_label):
        self.data = []
        for wnid, label in wnid_to_label.items():
            class_dir = os.path.join(emb_dir, wnid)
            if os.path.exists(class_dir):
                for file in os.listdir(class_dir):
                    if file.endswith('.pth'):
                        emb_path = os.path.join(class_dir, file)
                        self.data.append((emb_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        emb_path, label = self.data[idx]
        emb = torch.load(emb_path)
        return emb, label

# Model
class CBMPredictor(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CBMPredictor, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# Load datasets
train_dataset = EmbeddingDataset(train_emb_dir, wnid_to_label)
val_dataset = EmbeddingDataset(val_emb_dir, wnid_to_label)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, optimizer
model = CBMPredictor(embedding_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
best_acc = 0
for epoch in range(num_epochs):
    model.train()
    for embeddings, labels in train_loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Acc: {acc:.2f}%')

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'cbm_predictor.pth')

print(f'Best Val Acc: {best_acc:.2f}%')
print('Model saved as cbm_predictor.pth')