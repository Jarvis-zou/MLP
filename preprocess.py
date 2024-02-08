import pandas as pd
from sklearn.preprocessing import StandardScaler

new_onehot_data = pd.read_csv('data/onehot_data.csv')

onehot_label = new_onehot_data.iloc[:, 0]
energy_values_to_standardize = new_onehot_data.iloc[:, 1:]

# Standardizing the energy values
new_scaler = StandardScaler()
energy_values_standardized_new = new_scaler.fit_transform(energy_values_to_standardize)

energy_values_standardized_new_df = pd.DataFrame(energy_values_standardized_new, columns=new_onehot_data.columns[1:])
standardized_data = pd.concat([onehot_label, energy_values_standardized_new_df], axis=1)
standardized_data_csv_path = 'data/standardized_data.csv'
standardized_data.to_csv(standardized_data_csv_path, index=False)



