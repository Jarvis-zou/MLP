import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import MLP
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_data(file_path, bz, shuffle):
    data = pd.read_csv(file_path)
    labels = data.iloc[:, 0].apply(lambda x: list(map(int, x.split(','))))  # first column is Label
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    targets_tensor = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32)
    dataset = TensorDataset(labels_tensor, targets_tensor)

    return DataLoader(dataset, batch_size=bz, shuffle=shuffle)


# Create DataLoaders
train_set_path = "training_data/train_set.csv"
validation_set_path = "training_data/validation_set.csv"
test_set_path = "training_data/test_set.csv"

train_loader = load_data(train_set_path, bz=256, shuffle=False)
val_loader = load_data(validation_set_path, bz=256, shuffle=False)
test_loader = load_data(test_set_path, bz=256, shuffle=False)

# train
input_size = 49
hidden_size = 2048
output_size = 5
lr = 0.0003
momentum = 0.95

model = MLP(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def train_batch(model, labels, targets, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    outputs = model(labels)
    loss = criterion(outputs, targets)
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    loss = loss + 0.0001 * l1_norm
    loss.backward()
    optimizer.step()
    return loss.item()


def validate_batch(model, labels, targets, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(labels)
        loss = criterion(outputs, targets)
    return loss.item()


num_epoch = 200
train_losses = []
val_losses = []
for epoch in range(num_epoch):
    train_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epoch}")
    for labels, targets in train_loader_tqdm:
        loss = train_batch(model, labels, targets, criterion, optimizer)
        train_loss += loss
        train_loader_tqdm.set_postfix(Train_Loss=loss)
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    val_loader_tqdm = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epoch}")
    val_loss = 0
    for labels, targets in val_loader_tqdm:
        loss = validate_batch(model, labels, targets, criterion)
        val_loss += loss
        val_loader_tqdm.set_postfix(Val_Loss=loss)
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

torch.save(model, 'model.pth')

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('result.png')
plt.show()



