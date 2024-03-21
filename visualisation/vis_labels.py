import torch
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Assume the Data object has been correctly loaded into the variable 'data'
data_path = '../data/processed/data_homo.pt'
data = torch.load(data_path)
# Convert y to a Pandas DataFrame for easier visualization with Seaborn
# Assigning labels to the columns
labels = ['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence']
df = pd.DataFrame(data.y.numpy(), columns=labels)

# Creating the plots
fig, axs = plt.subplots(3, 2, figsize=(12, 8))

# Iterate through all indicators and plot histograms
for i, ax in enumerate(axs.flat):
    sns.histplot(df, x=labels[i], bins=20, kde=False, ax=ax)
    ax.set_title(f'{labels[i]} Distribution')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('results/labels_hist_zscore.png', dpi=300)