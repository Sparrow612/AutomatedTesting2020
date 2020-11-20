import matplotlib.pyplot as plt


def visualize(images, save_path):
    """数据可视化"""
    fig, ax = plt.subplots(
        nrows=5,
        ncols=5,
        sharex=True,
        sharey=True)

    ax = ax.flatten()
    for i in range(25):
        img = images[i].reshape(28, 28)
        ax[i].imshow(img, cmap='gray', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path)
