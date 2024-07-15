import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection

def brighten_colors(cmap, factor=0.5):
    c = cmap(np.arange(cmap.N))
    c[:, :-1] += (1.0 - c[:, :-1]) * factor
    c = np.clip(c, 0, 1)
    return ListedColormap(c)

def get_fold_colormap(fold):
    base_colormaps = [cm.plasma, cm.viridis, cm.inferno, cm.magma, cm.cividis]
    return brighten_colors(base_colormaps[fold % len(base_colormaps)])

# Plot Diagnostics
def plot_diagnostics(lonely_loss_train_collect, lonely_loss_val_collect, sentiment_loss_train_collect, sentiment_loss_val_collect, dice_loss_train_collect, dice_loss_val_collect, tversky_loss_train_collect, tversky_loss_val_collect, constrast_loss_train_collect, constrast_loss_val_collect, reason_loss_train_collect, reason_loss_val_collect):
    fig, axes = plt.subplots(6, 2, figsize=(30, 10))
    titles = ['Lonely Loss Train (per mini-batch)', 'Lonely Loss Val (per epoch)', 'Sentiment Loss Train (per mini-batch)', 'Sentiment Loss Val (per epoch)', 'Dice Loss Train (per mini-batch)', 'Dice Loss Val (per epoch)', 'Tversky Loss Train (per mini-batch)', 'Tversky Loss Val (per epoch)', 'Constrastive Loss Train (per mini-batch)', 'Constrastive Loss Val (per epoch)', 'Reason Loss Train (per mini-batch)', 'Reason Loss Val (per epoch)']
    data = [lonely_loss_train_collect, lonely_loss_val_collect, sentiment_loss_train_collect, sentiment_loss_val_collect, dice_loss_train_collect, dice_loss_val_collect, tversky_loss_train_collect, tversky_loss_val_collect, constrast_loss_train_collect, constrast_loss_val_collect, reason_loss_train_collect, reason_loss_val_collect]

    for ax, losses, title in zip(axes.ravel(), data, titles):
        patches = []

        for fold, loss in enumerate(losses, 1):
            x = np.arange(len(loss))
            y = np.array(loss)
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            colormap = get_fold_colormap(fold)
            norm = plt.Normalize(y.min(), y.max())
            lc = LineCollection(segments, cmap=colormap, norm=norm)
            lc.set_array(y)
            lc.set_linewidth(2)
            ax.add_collection(lc)

            color_for_legend = colormap(norm(y.min()))
            patches.append(Patch(color=color_for_legend, label=f'Fold {fold}'))

        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xlabel('Iterations', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, linestyle='--', linewidth=0.5)
        max_iterations = len(losses[0])
        tick_interval = max(1, max_iterations // 20)
        ax.set_xticks(np.arange(0, max_iterations + 1, tick_interval))
        ax.set_yticks(np.linspace(min(min(losses)), max(max(losses)), num=20))
        ax.autoscale_view()
        for _, spine in ax.spines.items():
            spine.set_linewidth(2)

        ax.legend(handles=patches, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.show()
