import pandas as pd

file_path = "data/raw_data.csv"
df = pd.read_csv(file_path)
labels = df["Label(A_A'_B_X)"]
energy_columns = df[['E_A1', "E_A'", 'E_P', 'E_K', 'E_H']]

element_list = set()
for label in labels:
    elements = label.split("_")[:-1]
    element_list.update(elements)

sorted_elements = sorted(element_list)


# Function to create one-hot encoding for a label
def label_to_onehot(label, elements_list):
    elements_in_label = label.split('_')[:-1]
    return [1 if element in elements_in_label else 0 for element in elements_list]


onehot_labels = [label_to_onehot(label, sorted_elements) for label in labels]
onehot_strings = [','.join(map(str, onehot)) for onehot in onehot_labels]

# Replacing the 'Label(A_A\'_B_X)' column in the original dataframe with these strings
df['Label(A_A\'_B_X)'] = onehot_strings
final_replaced_csv_file_path = 'data/onehot_data.csv'
df.to_csv(final_replaced_csv_file_path, index=False)