import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

# Load all metrics results
eval_files = glob.glob("results/*_eval.csv")
all_results = []
for file in eval_files:
    df = pd.read_csv(file)
    all_results.append(df)
results_df = pd.concat(all_results)

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Get model names without organization prefix for cleaner labels
results_df['model_name'] = results_df['model'].apply(lambda x: x.split('/')[-1])

# Set up bar positions
models = results_df['model_name'].unique()
x = np.arange(len(models))
width = 0.25

# Plot bars for each category
ax.bar(x - width, results_df['consistent_logit_acc'], width, label='Consistent', color='#2ecc71')
ax.bar(x, results_df['inconsistent_logit_acc'], width, label='Inconsistent', color='#e74c3c')
ax.bar(x + width, results_df['nonsense_logit_acc'], width, label='Nonsense', color='#3498db')

# Customize plot
ax.set_ylabel('Accuracy')
ax.set_title('Model Performance Across Categories (Logit)')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save plot
plt.savefig('results/model_comparison.png')
plt.close()

# similarly plot the exact match accuracy
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x - width, results_df['consistent_exact_match_acc'], width, label='Consistent', color='#2ecc71')
ax.bar(x, results_df['inconsistent_exact_match_acc'], width, label='Inconsistent', color='#e74c3c')
ax.bar(x + width, results_df['nonsense_exact_match_acc'], width, label='Nonsense', color='#3498db')

ax.set_ylabel('Accuracy')
ax.set_title('Model Performance Across Categories (Exact Match)')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('results/model_comparison_exact_match.png')
plt.close()
