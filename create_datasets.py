import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'data/onehot_data.csv'
data = pd.read_csv(file_path)
labels = data.iloc[:, 0]
energy = data.iloc[:, 1:]

# set ratios
train_ratio = 0.7
validation_ratio = 0.2
test_ratio = 0.1

# split dataset
X_train, X_temp, y_train, y_temp = train_test_split(labels, energy, test_size=1 - train_ratio, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42)

train_set = pd.concat([X_train, y_train], axis=1)
validation_set = pd.concat([X_val, y_val], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

train_set.to_csv('original_data_test/train_set.csv', index=False)
validation_set.to_csv('original_data_test/validation_set.csv', index=False)
test_set.to_csv('original_data_test/test_set.csv', index=False)
