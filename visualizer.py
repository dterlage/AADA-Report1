import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 6,6  # default hor./vert. size of plots, in inches
plt.rcParams['lines.markeredgewidth'] = 1  # to fix issue with seaborn box plots; needed after import seaborn

def ass_1(lines, ax=None):
    n, d, c = lines[0].split()
    print(n, d, c)
    lines.pop(0)
    df = pd.DataFrame([[float(j) for j in i.split()] for i in lines[:-int(c)]])
    cd = [[float(j) for j in i.split()] for i in lines[-int(c):]]
    if d == '2':
        if ax is None:
            ax = plt.gca()
        df.plot(kind='scatter', x=0, y=1, c=df.columns.size-1, colormap='tab20', colorbar=False, ax=ax)
        x, y = zip(*cd)
        ax.scatter(x, y, marker='X', s=150, c=range(int(c)), cmap='tab20', linewidths=1, edgecolors='black')
    if d == '3':
        df.columns = ['x', 'y', 'z', 'c']
        sns.pairplot(df, vars=['x', 'y', 'z'], hue='c')

def plot_one(fl):
    ass_1(read_plot_file(fl))
    plt.show()

def plot_two(fl1, fl2, results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    title1 = "Gonzalez: {score:.3f}, {time:.3f}s".format(score=results["Gonzalez"][0], time=results["Gonzalez"][1])
    title2 = "Kmeans++: {score:.3f}, {time:.3f}s".format(score=results["kmeans++"][0], time=results["kmeans++"][1])

    ass_1(read_plot_file(fl1), ax=axes[0])
    ass_1(read_plot_file(fl2), ax=axes[1])

    axes[0].set_title(title1)
    axes[1].set_title(title2)

    plt.tight_layout()
    plt.show()

def read_plot_file(fl):
    with open(fl) as f:
        lines = f.read().splitlines()
    a = lines[0].split()[0]
    print(a)
    lines.pop(0)
    #ass_1(lines)
    return lines