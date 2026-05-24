import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('experiments_results.csv')

df['distances'] = pd.to_numeric(df['distances'])
df['proportions'] = pd.to_numeric(df['proportions'])
df['dimension'] = df['dimension'].astype(str) + 'D'


# Create a combined label for algorithm + dimension
combo_label = df['algorithm'] + ' ' + df['dimension']
df['algorithm_dimension'] = combo_label

sns.set(style='whitegrid')

subset_dist = df[df['type'] == 0]
subset_pct = df[df['type'] == 1]

palette = {
    'Gonzalez 2D': '#8cc5e3',
    'Gonzalez 3D': '#3594cc',
    'kmeans++ 2D': '#f0b077',
    'kmeans++ 3D': '#ea801c',
}

# //
#     Total mean squared distance vs Distances and Proportions
# //

fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharey=False)

sns.lineplot(
    data=subset_dist,
    x='distances',
    y='score',
    hue='algorithm_dimension',
    markers=True,
    dashes=False,
    palette=palette,
    marker = 'o',
    ax=axes[0]
)
axes[0].set_title('Total mean squared distance vs Distances')
axes[0].set_xlabel('Distance')
axes[0].set_ylabel('Total mean squared distance')
axes[0].legend(title='Algorithm + Dimension')
axes[0].set_ylim(ymax=700000)  

sns.lineplot(
    data=subset_pct,
    x='proportions',
    y='score',
    hue='algorithm_dimension',
    markers=True,
    dashes=False,
    palette=palette,
    marker = 'o',
    ax=axes[1]
)
axes[1].set_title('total mean squared distance vs Proportions')
axes[1].set_xlabel('Proportion of Outliers (%)')
axes[1].set_ylabel('Total mean squared distance')
axes[1].legend(title='Algorithm + Dimension')
axes[1].set_ylim(ymax=700000)  
axes[1].set_xticks([5, 10, 25, 50, 75, 90, 95])  

plt.tight_layout()
plt.savefig('graphs/score.png', dpi=300)
print(f'Saved total mean squared distance plot')
plt.show()

# //
#     Mean squared distance divided by number of points vs Distances and Proportions
# //

fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharey=False)

sns.lineplot(
    data=subset_dist,
    x='distances',
    y='avg_score',
    hue='algorithm_dimension',
    markers=True,
    dashes=False,
    palette=palette,
    marker = 'o',
    ax=axes[0]
)
axes[0].set_title('Mean squared distance divided by number of points vs Distances')
axes[0].set_xlabel('Distance')
axes[0].set_ylabel('Mean squared distance divided by number of points')
axes[0].legend(title='Algorithm + Dimension')
axes[0].set_ylim(ymax=300)  

sns.lineplot(
    data=subset_pct,
    x='proportions',
    y='avg_score',
    hue='algorithm_dimension',
    markers=True,
    dashes=False,
    palette=palette,
    marker = 'o',
    ax=axes[1]
)
axes[1].set_title('Mean squared distance divided by number of pointsvs Proportions')
axes[1].set_xlabel('Proportion of Outliers (%)')
axes[1].set_ylabel('Mean squared distance divided by number of points')
axes[1].legend(title='Algorithm + Dimension')
axes[1].set_ylim(ymax=300)  
axes[1].set_xticks([5, 10, 25, 50, 75, 90, 95])  

plt.tight_layout()
plt.savefig('graphs/avg_score.png', dpi=300)
print(f'Saved total mean squared distance plot')
plt.show()


# //
#     Iterations vs Distances and Proportions
# //

fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharey=False)

sns.lineplot(
    data=subset_dist,
    x='distances',
    y='iterations',
    hue='algorithm_dimension',
    markers=True,
    dashes=False,
    palette=palette,
    marker = 'o',
    ax=axes[0]
)
axes[0].set_title('Iterations vs Distances')
axes[0].set_xlabel('Distance')
axes[0].set_ylabel('Iterations')
axes[0].legend(title='Algorithm + Dimension')
axes[0].set_ylim(ymax=80)  

sns.lineplot(
    data=subset_pct,
    x='proportions',
    y='iterations',
    hue='algorithm_dimension',
    markers=True,
    dashes=False,
    palette=palette,
    marker = 'o',
    ax=axes[1]
)
axes[1].set_title('Iterations vs Proportions')
axes[1].set_xlabel('Proportion of Outliers (%)')
axes[1].set_ylabel('Iterations')
axes[1].legend(title='Algorithm + Dimension')
axes[1].set_ylim(ymax=80)  
axes[1].set_xticks([5, 10, 25, 50, 75, 90, 95])  

plt.tight_layout()
plt.savefig('graphs/iterations.png', dpi=300)
print(f'Saved iterations plot')
plt.show()

