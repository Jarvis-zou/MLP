import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler


model = torch.load('model.pth')
model.eval()

test_data = pd.read_csv('training_data/test_set.csv')
selected_data = test_data.sample(n=1)

# load same StandardScaler
onehot_data = pd.read_csv('data/onehot_data.csv')
onehot_label = onehot_data.iloc[:, 0]
energy_values = onehot_data.iloc[:, 1:]
scaler = StandardScaler()
values_standardized = scaler.fit_transform(energy_values)

original_features = scaler.inverse_transform(selected_data.iloc[:, 1:])
original_features = torch.tensor(original_features, dtype=torch.float32)
label = selected_data.iloc[0, 0]
label_list = list(map(int, label.split(',')))
label_tensor = torch.tensor(label_list, dtype=torch.float32)

with torch.no_grad():
    predictions = model(label_tensor)

# recover label tensor to original element string label
with open("data/elements_tabel.txt", "r") as file:
    labels_str = file.readlines()

labels_str = [label.strip() for label in labels_str]
index_to_label = {i: label for i, label in enumerate(labels_str)}
original_labels = [index_to_label[i] + "_" for i, value in enumerate(label_tensor) if value == 1]
original_labels = original_labels[0] + original_labels[1] + original_labels[2] + original_labels[3]

original_predictions = scaler.inverse_transform(predictions.reshape(-1, 5))
print("Original Test Data:\n", original_labels, original_features)
print("Original Predictions:\n", original_predictions)

