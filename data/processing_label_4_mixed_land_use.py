import pandas as pd
import numpy as np

label_columns_list = ['office',	'sustenance', 'transport',	'retail', 'leisure', 'residence']
bike_labels = pd.read_csv('label/0-1_processed_merged_bikelocation_POI_filtered_wr.csv')
bus_labels = pd.read_csv('label/0-1_processed_merged_buslocation_POI_filtered_wr.csv')
tube_labels = pd.read_csv('label/0-1_processed_merged_trainlocation_POI_filtered_wr.csv')
combined_label = pd.concat([bike_labels, bus_labels, tube_labels], ignore_index=True)
label_columns_list = ['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence']

def shannon_diversity_index(row):
    data = row[row > 0]
    probabilities = data / data.sum()
    return -np.sum(probabilities * np.log(probabilities))

combined_label['shannon_diversity_index'] = combined_label[label_columns_list].apply(shannon_diversity_index, axis=1)

# 输出查看结果
quantile_25 = combined_label['shannon_diversity_index'].quantile(0.75)